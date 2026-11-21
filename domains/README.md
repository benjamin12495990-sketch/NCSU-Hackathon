# domains/

> **Status: concrete is built in the main app. `materials` (alloy hardness), `insulators` (dielectric
> constant) and `polymers` (glass transition) are live on the site; their folders hold the agent's
> `prepare.py` and `nn_generic.js`, copied byte-identical, plus `result.json` and `skeptic_check.json`.
> `chemistry` (solubility) has the same four files for a model trained and checked offline; its data
> file and weights are not in the repo and it is not on the site yet. The other folders are plans.** Sources were
> not checked again through MCPs (see docs/AGENTS.md), and the insulator element table was typed from
> memory by its agent and is uncited.

Scope beyond concrete. Each subfolder is one domain that reuses the project's method
(`../docs/SCOPE.md`): a surrogate model, a cautious lower estimate (on the live concrete page it covers 87.7% of test
results, short of the 90% aim), candidate generation by
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
