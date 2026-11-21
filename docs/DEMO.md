# Demo script: Formulation Designer

A five-minute live demo with a script you can read from. Every number below was read off the app
on 2026-11-20 by running the exact clicks listed, and the browser path is deterministic, so you
should see the same ones. If you see different numbers, something is set differently (see
"Reset to a known state").

Need three minutes? Cut Beat 5 and shorten Beat 6.

## Before you go on

- [ ] Open the site and let it load. "Design mixes" says **Loading model…** until the model is in.
- [ ] Use the deployed site (the README's API example uses `https://ncsu-hackathon.vercel.app`;
      check that is the live address) or `npm run dev` at <http://localhost:5173>.
- [ ] Browser zoom 110 to 125 percent so the back row can read the table. Pick light or dark
      with the switch at the top right and stay with it.
- [ ] Defaults showing: **25 MPa**, **28 days**, **General (nothing switched on)**, all four
      checkboxes ticked.
- [ ] Do not press **Ping server for a new batch** unless you are on the deployed site. It calls a
      Vercel route, so on `npm run dev` it shows an error. It also uses a fresh random seed, so its
      numbers will not match this script.
- [ ] If the projector has no WebGL the 3D box is replaced by a note. Everything else still works.

## Numbers to have in your head

| | 25 MPa, general | Marine, 35 MPa |
| --- | --- | --- |
| New candidate | **182** kg CO2e/m³ | **241** |
| Best mix a lab already tested | **192** | **228** |
| Best cement-only tested mix | **283** | **446** |
| Candidate vs best tested | **−5%** | **+6%** |
| Candidate vs cement-only | **−36%** | **−46%** |

Model: expected strength is off by **3.3 MPa** on average on **219** lab results from **62** mixes
it never trained on (R² 0.91). Data: **306** lab-tested mixes, Yeh (1998), UCI.

## The five minutes

### Beat 1: the problem (0:00 to 0:30). Say only, no clicking.

> Testing one concrete mix means casting it and waiting 28 days for it to cure. So labs only ever
> try a handful of the possible mixes. Cement is also where most of the CO2 in a mix comes from.
> We built a tool that proposes new mixes for a strength you need, ranks them by embodied CO2, and
> tells you which ones are worth spending a 28-day test on.

### Beat 2: the page (0:30 to 1:00)

**Do:** click the three bars at the top left. Point at the list. Press **Escape**.

> This is a menu of simulations. Seven families of materials. Four are live: concrete, alloys,
> insulators and polymers. The other three are marked Planned, and I'll show you what that looks
> like at the end. We'd rather show four things that work than seven that don't.

### Beat 3: design a mix (1:00 to 2:15)

**Do:** click **General use · 25 MPa**. Scroll down past the "Train a neural network" panel to the
result (you will use that panel in Beat 4).

The box of particles flies in over about two seconds, probably before you have scrolled to it. To
replay it while you talk, click the middle card ("Best mix a lab already tested"), then the top
card again. Each click rebuilds the box.

> One click. The app took the 306 mixes a lab has actually tested and varied them, binder amount,
> slag and fly ash share, water, to make 12,436 candidates. 39 already exist in the dataset, so it
> threw those out. 8,557 of the rest clear a cautious strength estimate of 25 MPa. It hid another
> 821 because no nearby lab result backs them.

**Point at the three cards on the right.**

> Top candidate: 182 kilograms of CO2 per cubic metre. The best mix a lab already tested for this
> strength is 192, so we're 5 percent better. Plain cement-only concrete is 283, so we're 36 percent
> better than that.

**Point at the 3D box.** The blue particles pulse.

> The pulsing blue ones are cement. It's half of this mix's CO2 with only 4 percent of its volume.
> The sizes and positions aren't to scale, and the page says so under the box.

**Point at the first table row.** Read "Neighbours passed" and "29, 39, 26 MPa measured".

> This candidate isn't tested. What we can check is its three nearest lab-tested mixes. They
> measured 29, 39 and 26 MPa. All three cleared 25, so it's marked Neighbours passed.

Be straight about the size of the win: **5 percent against the best tested mix is modest. The
36 percent is against plain cement.** Say which is which.

### Beat 4: can we trust it? (2:15 to 3:15)

**Do:** scroll back up to the form and click **Train the neural network** (next to Design mixes).
The page scrolls to the training panel just below the form and the curve starts drawing. It takes
about **7 to 8 seconds**. Talk over it. Run a design first, as in Beat 3: the comparison table
under the curve only appears when there is a result to compare.

> The designer uses tree models. To show the prediction isn't a black box, here's a neural network
> trained live in your browser from random numbers, on the 786 training results. The curve is its
> error on lab results it has never seen.

When it finishes:

> It went from 15.9 MPa off to 3.9 MPa off. The tree model we actually use is at 3.3, so the trees
> stay in charge. The network is a second opinion.

**Point at the generated-mix table under the curve.**

> On our top five candidates the network agrees they clear 25 MPa. For the top one it says 36
> where the trees say 30. Different numbers, same answer, and that's what you want from a check.

### Beat 5: a real brief (3:15 to 4:15)

**Do:** scroll up. In **What are you building?** choose **Marine or coastal**. Press **Design
mixes**. Scroll to the result.

> Now a harder brief. I said marine. The page changed the question on its own: strength raised from
> 25 to 35 MPa, and water to binder capped at 0.4, because seawater exposure falls under an ACI 318
> class that requires it. The table below shows the rule and how many sources it was seen in.

**Point at the three cards.**

> And here the tool tells you something unwelcome. Our best candidate is 241. A mix a lab already
> tested is 228. For this brief you'd just use the tested one. Against cement-only, 446, we're
> 46 percent lower. This is a triage tool. Sometimes the answer is "don't bother with a new mix",
> and it says so.

### Beat 6: how it can be wrong, and where it goes next (4:15 to 5:00)

**Do:** scroll to **How this can be wrong**. Leave it on screen.

> We put this box on the page on purpose. On results it never trained on, the strength estimate is
> off by 3.3 MPa on average. Our cautious estimate was supposed to be beaten by 9 lab results in 10
> and was beaten by 88 in 100, so it's less cautious than we intended, and it says that. Candidates
> are untested predictions. The more we search for the lowest CO2, the more likely we find a mix the
> model is too optimistic about. Cast it and test it before you rely on it.

**Do:** open the menu, choose **Corrosive solutions**.

> The same method needs a lab dataset to learn from. We haven't built this one, so the page says
> Planned, not built, and states what it would need first. Alloys, insulators and polymers each
> have a page where you press Train and watch a network learn; the numbers there come from files,
> with the trees and a guess of the average beside them.

> Questions?

### Optional beat: generate untested alloys (only if there is time, about 45 s)

**Do:** open the menu, choose **Alloys (hardness)**, scroll to **Generate untested alloys**. Leave the
target at 400 HV and the route at CAST. Press **Generate untested alloys**. It trains the network
first (about 2.5 s), then generates.

**Point at row 1** (Al28.9 Co10.0 Cr18.0 Fe24.2 Mo8.9 Ni10.0): predicted 773 HV, nearest tested alloy
605 HV, gap +168 HV.

> These are compositions nobody has made. The network predicts 773, but the closest alloy anyone
> actually measured is 605, so we show that gap on every row. Its typical miss is about 78 HV on
> alloys like the tested ones and about 102 on systems it never saw. The tested neighbours are the
> evidence; the prediction is a reason to test, not a result.

**Do:** slide the target to 800 HV and press Generate again. The page says nothing reaches it: the
best lower edge is about 597 HV.

> When it can't support a target, it says so instead of showing a list.

**Do not say** that any candidate was made, tested or measured, or that it is harder or better than
a tested alloy.

## Questions you will get

**Can I pour one of these?** No. They are predictions. Nearest tested neighbours agree, but the
mixes themselves are untested. The page says so on every result, and the last box says it again.

**Does it always find something better?** No. On the Structural preset (35 MPa) the top candidate
is 249 and a tested mix is 208, so the candidate is **20 percent worse** than what a lab already
has. At 25 MPa it was 5 percent better. Show it if asked: click **Structural · 35 MPa**.

**Why not every family?** Concrete and alloys have public measured datasets with citations; the
polymer values are literature measurements whose source we have not confirmed; the insulator
values are calculated by computer simulation (DFT) and the page says so. Solubility has a trained
model that is not on the site yet because its training file is too big for a phone. The method needs
real data for each family, and we would not fake the rest.

**Where do the CO2 numbers come from?** One fixed factor per ingredient, listed with sources at the
bottom of the page. Some are confirmed by a second source, some are labelled unconfirmed. Real
values change with plant, region and transport.

**Where do the environment rules come from?** ACI 318 as reported by secondary sources, not the
standard itself. The page says that and marks how many sources saw each rule.

**How wrong can the network be?** The training panel lists eight held-out results. One of them is a
3-day slag mix that measured 41.1 MPa and the network predicted 26.0, a 15 MPa miss. Early-age
results are its weak spot, and that is one more reason the trees are the main model.

**Does the network know chemistry?** No. After training, the panel runs a sweep test: on 26 tested
mixes, more water lowered its prediction on 23 and more age raised it on all 26, so it has picked up
both trends from the lab results. But on 6 of them it predicted a strength below zero at one day
old, which is impossible. It follows patterns; it does not understand them. Its 90% interval is
± 11.1 MPa, and on the held-out results that held for 93 in 100.

**Is the network as good as the trees?** On the page's held-out mixes the trees are ahead, 3.3
against 3.9 MPa. We also ran both on eight other held-out splits: an ensemble of five of these
networks averaged 4.2 MPa and the trees 4.3. The ensemble won four splits, lost three and tied one,
so we call it a tie. The trees' 3.3 comes from a split that suits them; on the other eight they
range from 3.7 to 5.3. The numbers are in `research/nn_vs_trees.json`.

**Can you make the network better than the trees?** We tried seven ideas tonight, each from a
different literature search, each re-run by a second agent. One survived: tick **Also give it the
textbook strength laws** and train again. It hands the network Abrams', Bolomey's and Feret's laws
as inputs. Over 16 held-out splits that lowered the network's error from 4.33 to 4.20 MPa and
halved its impossible below-zero predictions, but the trees average 4.28, so against the trees it is
still a tie. On the page's own split it shows 4.0 MPa against 3.9 without the laws: level. The
numbers are in `research/physics_inputs.json`. Do not say it beats the trees.

**What did AI do here?** Models predict strength. AI research agents compiled the materials
table, so its values are typical figures, not individually checked, and the page labels them
"AI judgment". The literature tools (SciSpace, Consensus, FastTrack, OpenAlex) were used to
trace and check the cited numbers. See `research/connectors.md` and the root README.

## Do not say

- "36 percent better than the best known mix." It is 36 percent against the best tested
  **cement-only** mix. Against the best tested mix it is 5 percent.
- "Validated" or "proven". Nothing generated has been cast.
- "Works for polymers and chemistry." Those are plans.
- That a generated alloy was made, tested, or is better than a tested alloy. They are predictions;
  the nearest tested alloy is the evidence.
- "It trains in three seconds." It took about 7.5 seconds on the machine used to write this.
- Anything about the materials shortlist ranking being right. Its humidity and wear ratings are
  AI judgments, and the page says most of its values are typical figures, not individually
  checked. In the marine run its top three were UHPC, alumina and silicon carbide. Skip that table
  in the demo.

## Reset to a known state

Reload the page. If a number still differs, check: strength 25, age 28 days, **What are you
building?** on General, and all four checkboxes ticked ("Cautious" and "Only mixes backed by
nearby lab results" in particular). The browser path is seeded, so the same inputs give the same
numbers every time.
