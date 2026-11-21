# domains/

> **Status: only concrete is built.** Every domain in this folder is a plan. None has a
> dataset, a trained model or any references yet.

Scope beyond concrete. Each subfolder is one domain that reuses the project's method
(`../docs/SCOPE.md`): a surrogate model, a calibrated cautious bound, candidate generation by
perturbing known-good examples, ranking by a cost objective, and a "neighbours passed" check.

**Chemistry and materials only. No biology** — it is a complete subject of its own and is
deliberately excluded.

- `_TEMPLATE/` — copy this to start a new domain.

## Domains
- **polymers/** — carbon-based polymers (Tg, strength, thermal stability)
- **corrosives/** — corrosive solutions & material compatibility
- **cleaning/** — cleaning products, surfactants, detergents
- **insulators/** — thermal & electrical insulators / dielectrics
- **materials/** — general materials properties (concrete is the first worked example)
- **chemistry/** — general molecular / mixture properties

### Reach domains (same engine, later)
coatings · adhesives · lubricants · electrolytes & battery chemistries · refrigerants · dyes &
pigments · catalysts · propellants · sealants · composites · glasses & ceramics.

Both model scripts (`train/train.py` trees, `pipeline/train_nn.py` neural net) were written for
concrete and could be pointed at another tabular dataset. None of these domains has one yet. Literature grounding for every domain lives in `../research/references/<domain>.md`.
