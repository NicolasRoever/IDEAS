# Standard Idea Tests

Use this document to calibrate brainstorming, feedback, and idea generation. It encodes my research aesthetic, recurring objections, and what I consider a strong vs. weak idea. When I ask you to brainstorm or evaluate a research idea, apply these filters.

---

## 1. My recurring objections (apply these as kill tests)

### "This is just psychology, not economics."
If the result is "people behave differently under treatment X," that's a psych finding. I need a *market*, *allocation*, or *welfare* consequence. Pure behavioral patterns without equilibrium reasoning are insufficient.
- *Example*: "Students who get AI career advice explore more careers" — psychology. "AI career advice changes sorting in a matching market with capacity constraints" — economics.
- *Example*: "People with stronger professional identity resist AI more" — psychology. But if you can show this creates an aggregate misallocation with a quantifiable welfare cost — economics.

### "This is obvious / doesn't pass the surprise test."
Would a smart economics seminar audience be surprised by the finding? If not, it's not a paper. Confirming what everyone already believes, even with clean causal identification, is a weak contribution.
- *Example*: "Tired people make worse decisions with AI" — obviously true, no paper.
- *Example*: "AI career tools help everyone" — obvious prior. But "AI career tools *worsen* sorting for a specific, identifiable reason" — that's surprising.

### "This is just [existing paper] in a new domain."
- *Example*: "Misperceived norms about career prestige" = Bursztyn/González/Yanagizawa-Drott in a career setting. Referee says: what's new?
- *Example*: "Motivated beliefs about AI ability" = Zimmermann with a new signal source. What's the *theoretical* result that surprises someone who already knows Zimmermann?

### "The assumption is wrong / unrealistic."
I test whether the premise of the paper actually holds before building on it. If the premise is empirically false, the whole structure collapses.
- *Example*: "Nobody would prefer a world without AI" — so any framework built on people wanting to collectively reject AI is dead on arrival.
- *Example*: "Everybody knows everybody else is using AI" — so pluralistic ignorance about AI usage is an implausible premise.
- *Example*: "Meta-cognitive skill is probably just highly correlated with IQ" — so a paper claiming meta-cognitive skill is a novel, unobserved dimension is probably wrong.
- *Example*: "Many consultants actually report high job satisfaction" — so a paper built on insiders hating their jobs needs real data, not assumptions.

### "Heterogeneity results are not papers."
A finding that works "especially for subgroup X" is not a punchline. I want a main effect or a structural prediction. Treatment effect heterogeneity is supplementary evidence, not a contribution. 
- *Example*: "AI tools widen mismatch inequality because they help high-metacognition types but not low types" — this is a heterogeneity story. I explicitly do not want this.

### "Simple information provision is boring."
"Give people information → they update" is trivial. Information experiments need a twist — a market mechanism, a non-obvious interaction, or a welfare reversal.
- *Example*: "Students lack info about jobs → give info → they diversify" — done, boring.

### "This will be transient / dated in two years."
Anything tied to the *current* state of AI capability (e.g., algorithmic aversion, current sycophancy behavior) will be obsolete when AI improves. The mechanism needs to be structural, not contingent on today's technology.
- *Example*: "People distrust AI career advice" — this will change. No paper.
- *Example*: "AI is sycophantic" — models will be developed to push back. Transient.


### "Too many degrees of freedom / no clean identification."
I want experiments with ONE main outcome, pre-registerable, where the result is either there or it isn't. Multi-outcome measurement infrastructure projects with lots of researcher discretion are weak.

---

## 2. Practical constraints

- **Budget**: Limited. Multi-domain large-scale experiments are usually out of reach. I prefer designs that are cheap to run (Prolific, API calls for AI interactions, single-domain studies).
- **Platform**: Typically Prolific or student samples (~1000 participants).
- **Publication target**: Top-5 economics journals (QJE, AER, Econometrica, JPE, REStud) or strong field journals.
- **Existing work**: I'm a third-year PhD student supervised by Chris Roth. Co-authors include Bursztyn and Haaland. My comparative advantage is combining technical infrastructure (Python, SQL, APIs) with rigorous experimental design.

---

## 3. Illustrative examples of ideas I rejected and why

| Idea | My objection | Category |
|------|-------------|----------|
| "Collective trap in AI adoption: nobody wants to admit using AI" | "Not realistic. Everyone uses AI, and everyone knows everyone uses it." | Assumption is wrong |
| "Impostor syndrome externality from secret AI use" | "Seems untrue and non-relevant." | Assumption is wrong |
| "Preference falsification about AI in organizations, Kuran-style cascades" | "Far-fetched." | Implausible mechanism |
| "Identity-driven misallocation: high-identity professionals resist AI" | "Too much pure psychology." | Not economics |
| "Students lack info about careers → give info → they diversify" | Trivial information provision | Obvious / boring |
| "Misperceived career norms among students" | "Bursztyn et al. in a new domain = replication" | Not novel |
| "Algorithmic aversion in career domains" | "Transient. The status quo will change rapidly." | Will be dated |
| "Social comparison after career test results" | "Pure psychology." | Not economics |
| "Generation effect in career reflection" | "Pure psychology." | Not economics |
| "AI sycophancy widens mismatch inequality (helps high-meta types, hurts low)" | "This is a heterogeneity result. I DO NOT WANT THAT." | Heterogeneity ≠ paper |
| "Meta-cognitive skill as novel unobserved labor dimension" | "Probably just highly correlated with IQ." | Assumption is wrong |
| "Premature stopping of career exploration with AI" | "AI could be really good, so premature stopping could be optimal." | Assumption is wrong |
| "WTP for more precise career test results" | "With AI, info is so cheap — everybody will take it." | Dissolves with AI |
| "Multi-domain measurement of preference instability" | "Not a lot of money. And dwells on a quite obvious conceptual point." | Too expensive + obvious |
| "Endogenous lemons in market for expertise" (when asked for key objections) | "Reputation solves this. Competition among AI-augmented experts may just work. Where exactly does unraveling happen in practice?" | Self-healing mechanisms |
| "AI as ability revelation / Dunning-Kruger correction" | "Where's the formal novelty beyond Zimmermann + a new signal? The punchline might be obvious." | Not novel enough |
