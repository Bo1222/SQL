"""SQLite AST requirements bound to real schema and SQLGlot scopes.

Qualification expands USING/NATURAL and projection stars on a COPY for checking.
The original SQL is never rewritten for execution. Unsupported lineage is quarantined.
"""
from functools import lru_cache
from pathlib import Path
import sqlite3

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope

PARSER_VERSION = "30.30.0"
AST_RULE_VERSION = "sqlite-scoped-v2"


class Unresolved(ValueError):
    pass


@lru_cache(maxsize=128)
def sqlite_schema(db_path):
    path = Path(db_path).resolve(strict=True)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        conn.execute("PRAGMA query_only=ON")
        schema = {}
        for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
            quoted = table.replace('"', '""')
            schema[table.lower()] = {r[1].lower(): r[2] or "TEXT"
                                     for r in conn.execute(f'PRAGMA table_xinfo("{quoted}")') if r[6] != 1}
    return schema


def extract(sql, db_path=None, schema=None):
    result = {"parser": "sqlglot", "parser_version": sqlglot.__version__,
              "rule_version": AST_RULE_VERSION, "ok": False, "tables": [],
              "columns": [], "joins": [], "table_instances": [], "reasons": []}
    try:
        if sqlglot.__version__ != PARSER_VERSION:
            raise Unresolved("parser_version_mismatch")
        statements = sqlglot.parse(sql, read="sqlite", error_level=sqlglot.ErrorLevel.RAISE)
        if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.SetOperation)):
            raise Unresolved("expected_one_readonly_query")
        parsed = statements[0]
        if any(w.args.get("recursive") for w in parsed.find_all(exp.With)):
            raise Unresolved("recursive_cte_not_supported")
        # SQLite permits ambiguous duplicate output names, whose lineage is not safe here.
        schema = schema if schema is not None else sqlite_schema(str(db_path))
        schema = {t.lower(): {c.lower(): typ for c, typ in cols.items()} for t, cols in schema.items()}
        if any("." in t or any("." in c for c in cols) for t, cols in schema.items()):
            raise Unresolved("dotted_schema_identifiers_not_supported")
        checked = qualify(normalize_identifiers(parsed.copy(), dialect="sqlite"),
                          dialect="sqlite", schema=schema,
                          allow_partial_qualification=False, validate_qualify_columns=True)
        scopes = list(traverse_scope(checked))
        scope_id = {id(scope): f"s{index}" for index, scope in enumerate(scopes)}
        instances, tables, columns, joins = {}, set(), set(), []
        for scope in scopes:
            for alias, (_, source) in scope.selected_sources.items():
                if isinstance(source, exp.Table):
                    if source.db or source.catalog or source.name.lower() not in schema:
                        raise Unresolved(f"unknown_table:{source.sql()}")
                    table = source.name.lower()
                    instance = f"{scope_id[id(scope)]}:{alias}"
                    instances[(id(scope), alias)] = {"id": instance, "table": table,
                                                     "alias": alias, "scope": scope_id[id(scope)]}
                    tables.add(table)

        def bind(scope, column, visited=None):
            if column.is_star:
                raise Unresolved("unexpanded_column_star")
            visited = set() if visited is None else set(visited)
            name, alias = column.name.lower(), column.table.lower()
            owner = scope
            while owner is not None and alias not in owner.selected_sources:
                if not owner.can_be_correlated:
                    break
                owner = owner.parent
            if owner is None or alias not in owner.selected_sources:
                raise Unresolved(f"unbound_column:{column.sql()}")
            _, source = owner.selected_sources[alias]
            if isinstance(source, exp.Table):
                table = source.name.lower()
                if name not in schema[table]:
                    raise Unresolved(f"unknown_column:{table}.{name}")
                item = instances[(id(owner), alias)]
                return {(table, name, item["id"])}
            if not isinstance(source, Scope):
                raise Unresolved("unsupported_source")
            key = (id(source), name)
            if key in visited:
                raise Unresolved("recursive_column_lineage")
            visited.add(key)
            if isinstance(source.expression, exp.SetOperation):
                raise Unresolved("set_operation_output_lineage_not_supported")
            projections = [p for p in source.expression.selects if p.alias_or_name.lower() == name]
            if len(projections) != 1:
                raise Unresolved(f"ambiguous_derived_output:{name}")
            projection = projections[0]
            refs = []
            for col in projection.find_all(exp.Column):
                if col.find_ancestor(exp.Select) is source.expression:
                    refs.append(col)
            if any(isinstance(n, (exp.Subquery, exp.Select)) for n in projection.walk()):
                raise Unresolved("scalar_subquery_output_lineage_not_supported")
            return set().union(*(bind(source, c, visited) for c in refs)) if refs else set()

        def direct_bind(scope, column, visited=None):
            # Dependency lineage can be many-to-one through an expression; a
            # relationship endpoint must instead be a direct renamed column.
            visited = set() if visited is None else set(visited)
            alias, name, owner = column.table.lower(), column.name.lower(), scope
            while owner is not None and alias not in owner.selected_sources:
                if not owner.can_be_correlated:
                    break
                owner = owner.parent
            if owner is None or alias not in owner.selected_sources:
                raise Unresolved("join_endpoint_not_bound")
            _, source = owner.selected_sources[alias]
            if isinstance(source, exp.Table):
                return bind(owner, column)
            if not isinstance(source, Scope) or isinstance(source.expression, exp.SetOperation):
                raise Unresolved("join_endpoint_scope_not_supported")
            key = (id(source), name)
            if key in visited:
                raise Unresolved("cyclic_join_endpoint")
            visited.add(key)
            projections = [p for p in source.expression.selects if p.alias_or_name.lower() == name]
            if len(projections) != 1:
                raise Unresolved("ambiguous_join_endpoint")
            projection = projections[0]
            projection = projection.this if isinstance(projection, exp.Alias) else projection
            if not isinstance(projection, exp.Column):
                raise Unresolved("computed_join_output_lineage_not_supported")
            return direct_bind(source, projection, visited)

        for scope in scopes:
            for col in scope.columns:
                if col.is_star:
                    # COUNT(table.*) is not SQLite; COUNT(*) has no Column node.
                    raise Unresolved("unexpanded_column_star")
                for table, name, instance in bind(scope, col):
                    columns.add(f"{table}.{name}")
            # Every cross-instance comparison in this scope is a relationship.
            # Keep self-join instances distinct; complex/non-equality JOINs quarantine.
            for comparison in scope.expression.find_all(exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE):
                if comparison.find_ancestor(exp.Select) is not scope.expression:
                    continue
                if not isinstance(comparison.this, exp.Column) or not isinstance(comparison.expression, exp.Column):
                    cross_instances = set()
                    for col in comparison.find_all(exp.Column):
                        if col.find_ancestor(exp.Select) is scope.expression:
                            cross_instances.update(x[2] for x in bind(scope, col))
                    if comparison.find_ancestor(exp.Join) or len(cross_instances) > 1:
                        raise Unresolved("expression_join_not_supported")
                    continue
                left = direct_bind(scope, comparison.this)
                right = direct_bind(scope, comparison.expression)
                if len(left) != 1 or len(right) != 1:
                    raise Unresolved("computed_join_lineage_not_supported")
                a, b = next(iter(left)), next(iter(right))
                if a[2] == b[2]:
                    continue
                if not isinstance(comparison, exp.EQ):
                    raise Unresolved("non_equality_join_not_supported")
                joins.append({"scope": scope_id[id(scope)],
                              "left": {"table": a[0], "column": a[1], "instance": a[2]},
                              "right": {"table": b[0], "column": b[1], "instance": b[2]},
                              "operator": "=", "condition": comparison.sql(dialect="sqlite")})
            # IN(subquery) is a semi-join. Preserve its cardinality semantics,
            # rather than pretending it is an ordinary inner join.
            for membership in scope.expression.find_all(exp.In):
                if membership.find_ancestor(exp.Select) is not scope.expression:
                    continue
                query = membership.args.get("query")
                if query is None:
                    continue
                if not isinstance(membership.this, exp.Column):
                    raise Unresolved("computed_membership_join_not_supported")
                query = query.this if isinstance(query, exp.Subquery) else query
                inner = next((s for s in scopes if s.expression is query), None)
                if inner is None or not isinstance(query, exp.Select) or len(query.selects) != 1:
                    raise Unresolved("membership_output_not_resolved")
                projection = query.selects[0]
                projection = projection.this if isinstance(projection, exp.Alias) else projection
                if not isinstance(projection, exp.Column):
                    raise Unresolved("computed_membership_output_not_supported")
                left, right = direct_bind(scope, membership.this), direct_bind(inner, projection)
                if len(left) != 1 or len(right) != 1:
                    raise Unresolved("membership_lineage_not_resolved")
                a, b = next(iter(left)), next(iter(right))
                joins.append({"scope": scope_id[id(scope)],
                              "left": {"table": a[0], "column": a[1], "instance": a[2]},
                              "right": {"table": b[0], "column": b[1], "instance": b[2]},
                              "operator": "NOT IN" if isinstance(membership.parent, exp.Not) else "IN",
                              "condition": membership.sql(dialect="sqlite")})
            # A connected path is required for multi-source queries; explicit CROSS is
            # allowed structurally but flagged for the mandatory semantic review.
            sources = list(scope.selected_sources)
            if len(sources) > 1:
                for join in scope.expression.args.get("joins") or []:
                    if join.args.get("on") is not None and not list(join.args["on"].find_all(exp.Column)):
                        raise Unresolved("join_without_column_relationship")
                # No invented FK edges; disconnected joins are sent for verification.
                local = [instances[(id(scope), a)]["id"] for a in sources if (id(scope), a) in instances]
                connected = {local[0]} if local else set()
                changed = True
                while changed:
                    changed = False
                    for j in joins:
                        a, b = j["left"]["instance"], j["right"]["instance"]
                        if (a in connected) != (b in connected):
                            connected.update([a, b]); changed = True
                if any(i not in connected for i in local):
                    raise Unresolved("disconnected_or_cross_join_requires_manual_resolution")
        # Stars remaining under projections are unsupported; COUNT(*) is exempt.
        for star in checked.find_all(exp.Star):
            if not isinstance(star.parent, exp.Count):
                raise Unresolved("unexpanded_star")
        unique_joins = {str(j): j for j in joins}
        result.update(ok=True, tables=sorted(tables), columns=sorted(columns),
                      joins=list(unique_joins.values()), table_instances=list(instances.values()),
                      checked_sql=checked.sql(dialect="sqlite"))
    except Exception as exc:
        result["reasons"] = [f"{type(exc).__name__}:{exc}"]
    return result
