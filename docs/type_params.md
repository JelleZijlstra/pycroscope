# Type Parameter Scope Design

This note sketches the intended shape of `pycroscope/type_params.py`. The goal is
to make ownership of type parameters explicit without scattering owner-binding,
aliasing, and annotation-validity rules through `name_check_visitor.py`.

## Terms

A **type parameter** is one of pycroscope's canonical `TypeParam` values:
`TypeVarParam`, `ParamSpecParam`, or `TypeVarTupleParam`.

A **type parameter owner** is the class, function, or type alias definition that
binds a type parameter. Owners are represented by `TypeParamOwner`.

A **type parameter identity** is the thing used to recognize a type parameter in
source or runtime annotations. It should be restricted to:

```python
TypeParamIdentity = TypeVarLike | ast.AST
```

Use a `TypeVarLike` identity when there is a runtime `TypeVar`, `ParamSpec`, or
`TypeVarTuple` object. Use an AST identity when pycroscope synthesized the type
parameter while analyzing unimportable code and no runtime object exists. Avoid
using arbitrary `object` identities; every identity should be explainable as
either the runtime object or the syntax node that declares the parameter.
Avoid manually creating `TypeVarLike` objects at type checking time.

## Proposed Model

`ActiveTypeParams` should be an explicit stack of nested type-parameter scopes.
Each scope owns the type parameters declared by one definition, plus the small
amount of state needed to validate references to those parameters while visiting
annotations.

```python
@dataclass
class TypeParamScope:
    owner: TypeParamOwner | None
    bindings: dict[TypeParamIdentity, TypeParam]
    disallowed: set[TypeParamIdentity]
```

### `owner`

The owner for type parameters declared directly in this scope. It is `None` only
for helper scopes that do not declare parameters, such as a temporary annotation
validation scope.

This replaces caller-side owner reconstruction. Code that visits a PEP 695
`TypeVar` node should ask `ActiveTypeParams` for the current declaration owner
instead of duplicating class/function/alias owner logic in `NameCheckVisitor`.

### `bindings`

Maps every known identity for a type parameter to its canonical `TypeParam`.
Usually this contains one identity: `type_param.typevar`. During import-failure
or synthetic-class handling, the same canonical parameter may also need an AST
identity or a pre-rebinding runtime identity. Those aliases should be recorded in
this mapping instead of passed around as parallel `additional_identities` lists.

The important invariant is that lookup returns the already owner-bound canonical
parameter. Callers should not need to call `with_type_param_owner()` after a
parameter has entered a scope.

### `disallowed`

A set of identities that are lexically visible but invalid in the current
annotation context. This is for rules such as "an outer class type parameter is
not valid in this nested alias annotation" or other contexts where normal name
resolution can find a type parameter but the typing spec does not permit using
it there.

This should stay as scope state because it is about validity of a reference at a
particular point in the syntax, not about the parameter's owner or canonical
identity.

## Why There Is No `kind` Field

The scope does not need a `kind` field if callers provide the owner explicitly
when entering the definition. A field such as `"class"`, `"function"`, or
`"alias"` would duplicate information that is already represented by the owner
type (`ClassOwner` or runtime class, `FunctionOwner`, `AliasOwner`).

If a future rule truly depends on the syntactic kind of a scope instead of the
owner, that should be modeled as a narrow field for that rule. It should not be
part of the core owner-binding abstraction.

## Binding API

The main operations should be:

```python
def push_scope(owner: TypeParamOwner | None = None) -> AbstractContextManager[None]:
    ...

def declare(type_param: TypeParam, aliases: Iterable[TypeParamIdentity] = ()) -> TypeParam:
    ...

def lookup(identity: TypeParamIdentity) -> TypeParam | None:
    ...

def bind_all(type_params: Sequence[TypeParam], owner: TypeParamOwner) -> TypeParamBindingResult:
    ...
```

`declare()` should owner-bind the parameter using the current scope owner, record
all aliases in `bindings`, and return the canonical parameter.

`bind_all()` should replace `NameCheckVisitor._bind_type_param_owners()`. It
should return:

```python
@dataclass(frozen=True)
class TypeParamBindingResult:
    type_params: tuple[TypeParam, ...]
    substitutions: TypeVarMap
    aliases: tuple[frozenset[TypeParamIdentity], ...]
```

### `type_params`

The canonical, owner-bound parameters.

### `substitutions`

A map from the incoming unbound parameters to the canonical parameters. This is
needed to rewrite bounds, constraints, defaults, function annotations, alias
values, and class bases that mention the old parameter objects.

### `aliases`

For each canonical parameter, the identities that should also resolve to it.
This replaces ad hoc dictionaries and parallel alias lists in class registration.

## Responsibilities

`type_params.py` should own:

- Owner binding for classes, functions, aliases, and runtime generic objects.
- Lookup from `TypeParamIdentity` to canonical `TypeParam`.
- Alias identities created while rebinding synthetic or runtime parameters.
- Validity checks for direct type-parameter use in annotation contexts.
- Legacy TypeVar rejection state.

`name_check_visitor.py` should still own:

- Building the correct `TypeParamOwner` for a class, function, or alias node.
- Deciding when to enter and exit scopes while visiting the AST.
- Reporting errors through the existing visitor methods.

Variance collection can remain in `type_params.py`, but it should be separated
from owner-binding state. It is usage-analysis state, not identity or ownership
state.

## Migration Plan

1. Add the typed identity alias and make all type-parameter lookup APIs accept
   `TypeParamIdentity` instead of `object`.
2. Make `TypeParamScope.bindings` the source of truth and remove the parallel
   `_active_pep695_identities` and `_active_pep695_type_params` stacks.
3. Move owner binding and type-parameter component rewriting from
   `NameCheckVisitor` into a `bind_all()` helper in `type_params.py`.
4. Replace `additional_identities` with explicit aliases passed to `declare()`
   or returned by `bind_all()`.
5. Once every construction path has an owner, include owners in equality and
   hashing.
