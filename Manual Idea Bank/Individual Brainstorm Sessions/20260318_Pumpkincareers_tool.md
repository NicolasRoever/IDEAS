# Monetizing unpaid student survey data

**Created:** 3/18/2026 9:27:11  
**Updated:** 3/18/2026 9:57:00  
**Exported:** 3/18/2026 10:42:50  
**Link:** [https://claude.ai/chat/63a41d91-0718-4715-a60c-dd93518ad6a4](https://claude.ai/chat/63a41d91-0718-4715-a60c-dd93518ad6a4)  

## Prompt:
18/03/2026, 09:27:12

I have the following setting. 

I can field a survey to ~1000 business economics students. However, I do not pay them, so the survey needs to provide some value (it is advertised on Instagram and within a career coaching program). Can I do something with this. Be broad, suggest ideas, beased on \research-idea-evaluator.



## Response:
18/03/2026, 09:28:30

Let me read the skill file first to ground my feedback in your research taste.

````plaintext
View: Read the research idea evaluator skill
````

````plaintext
View: Read research profile
````

````plaintext
View: Check skill assets
````

````plaintext
View: Read idea testing criteria
````

````plaintext
View: Read research vision
````

Good, now I have full context on your research taste, kill tests, and practical setup. Let me think through this carefully.

Here's the landscape: ~1,000 business economics students, unpaid, recruited via Instagram and a career coaching program (pumpkincareers, presumably). The constraint that they're unpaid but get value means the survey itself needs to function as a service — career feedback, self-knowledge, or something they'd actually want to do.

Let me brainstorm broadly, then run each idea through your filters.

---

**Idea 1: Demand for AI therapy as a function of stigma — using the career coaching pipeline as entry point**

You give students a validated anxiety/stress screener as part of the "career coaching" survey. Then you randomize an offer: access to an AI-CBT session (Sonia) vs. a referral to the university counseling center vs. both. The outcome is take-up. The value proposition to students is real: they get a mental health screening + access to a free tool.

The punchline would be about *revealed demand* for mental health services when you remove stigma barriers — connecting directly to your AI-CBT work showing AI is low-barrier and doesn't engender shame. The economics angle: if AI lowers the stigma cost, you can estimate the "stigma tax" on mental health service utilization by comparing take-up rates.

*Kill test check:* This is close to your existing work and the punchline ("removing stigma increases take-up") might be obvious. It might fail the surprise test. Also, it's more health economics than your core comparative advantage. **Verdict: probably too incremental on your existing AI-CBT line, but worth noting as a pilot.**

---

**Idea 2: Eliciting career preferences before and after structured reflection — testing whether people's stated preferences are "wrong" by their own lights**

This connects to your welfare/preference elicitation project. Students take a career preference survey (e.g., ranking sectors, salary vs. meaning tradeoffs, risk tolerance). Then half are randomized into a Socratic AI conversation that pushes them to reflect on *why* they hold these preferences. Afterward, everyone re-states preferences. The value to students: genuine career self-reflection, which is exactly what pumpkincareers would want to offer.

The punchline: How much do career preferences move after reflection, and do people *endorse* the change (i.e., do they say "my new preferences are more truly mine")? This is a direct empirical test of the Bernheim-Rangel framework — are pre-reflection preferences welfare-relevant, or are post-reflection preferences? And the adjudication mechanism (asking people which version of themselves they endorse) is the novel methodological contribution.

*Kill test check:* This is theory-driven (Bernheim-Rangel, experienced vs. remembered utility). The surprise could be that people's career preferences are *massively* unstable under reflection — or that they're remarkably stable and people endorse their initial gut. Either direction is interesting. The worry is "this is just psychology" — but if you can connect preference instability to misallocation in a matching market (students sorting into wrong careers), it becomes economics. The identification is clean: one main outcome (preference distance pre vs. post), pre-registerable. **Verdict: This is your strongest candidate. It directly advances your APE project with real data.**

---

**Idea 3: Social desirability bias in career aspirations — using your methodological expertise**

You measure career aspirations using a standard direct question, then randomize students into a list experiment or indirect elicitation for sensitive career preferences (e.g., "I'm primarily motivated by money," "I want to work in a prestigious firm mainly for status," "I plan to leave Germany for higher pay"). The value to students: a personalized career motivation profile.

The punchline: Career aspirations are contaminated by social desirability — students overstate intrinsic motivation and understate pecuniary motivation. The gap between stated and revealed (via indirect methods) preferences has *allocative consequences*: if career counselors take stated preferences at face value, they systematically steer students toward the wrong jobs.

*Kill test check:* This leverages your Social Desirability Atlas directly. The "just psychology" risk is real — you need the allocative/welfare consequence to make it economics. The novelty question: is it just "social desirability bias exists in domain X"? You'd need a sharper twist. Perhaps: the size of the social desirability gap *predicts* actual career dissatisfaction 1-2 years later (if you can do a follow-up). Or: the gap is largest for the highest-ability students, creating the worst misallocation where it matters most. **Verdict: Decent but needs a sharper punchline beyond "SDB exists in careers."**

---

**Idea 4: Causal models and career choice — do students have wrong theories about what makes careers satisfying?**

Inspired by your interest in how people's own causal models drive decisions. You elicit students' *theories* about what predicts career satisfaction (salary? autonomy? prestige? match quality?). Then you confront them with actual longitudinal evidence on career satisfaction predictors. The randomization: half get the evidence, half don't. Then measure how they update their career plans.

The punchline: People have systematically wrong causal models about career satisfaction — they overweight salary and prestige, underweight autonomy and match. And these wrong models are *self-reinforcing* because the people who chose based on salary never learn the counterfactual. This is a market failure story: incorrect causal models create persistent misallocation that doesn't self-correct because of missing counterfactual information.

*Kill test check:* "Simple information provision is boring" — this is dangerously close. The twist would need to be about *why* people hold wrong causal models (narrative reasoning? social proof?) and whether correcting the model actually changes behavior. Also, "people overweight salary" might fail the surprise test. **Verdict: interesting conceptual setup but the information provision concern is serious. Needs a mechanism beyond "give info → people update."**

---

**Idea 5: Measuring the "narrative premium" — how much do career narratives distort choices?**

Students evaluate identical career profiles that differ only in the narrative framing (e.g., "management consulting: you solve complex strategic problems for Fortune 500 companies" vs. the same job described factually: "you make PowerPoint decks and do data entry for 70 hours a week"). You measure WTP (in terms of salary sacrifice) for the narrative vs. factual description. The value to students: understanding how susceptible they are to career marketing.

The punchline: There is a quantifiable "narrative premium" — students will accept significantly lower pay for narratively-attractive descriptions of identical jobs. This premium is a *market inefficiency*: firms that invest in career narratives extract rents from workers. The welfare question is whether this is manipulation or legitimate preference (people might genuinely value working in a job they can tell a good story about).

*Kill test check:* This is economics — there's a price, a market distortion, and welfare implications. The surprise test: is it obvious that narratives matter? Yes, somewhat. But quantifying the premium and showing it's large enough to matter for allocation could be surprising. The transience test: narratives aren't going away, this is structural. **Verdict: Interesting but might be seen as "framing effects in a new domain" — needs a structural/theoretical hook to distinguish it from behavioral psychology.**

---

My ranking for you, applying your full filter set:

**Idea 2 (preference elicitation + reflection) is the clear frontrunner.** It's theory-driven (Bernheim-Rangel), methodologically novel (the adjudication mechanism), directly advances your APE project, fits the sample perfectly, and provides genuine value to participants. The main risk is that you need to make the "economics, not psychology" case — but if you frame it around misallocation in labor market sorting, it works.

**Idea 3 (social desirability in career aspirations)** is second — it leverages your existing methodological expertise and the sample is ideal, but it needs a sharper punchline.

**Ideas 4 and 5** have interesting conceptual cores but face serious kill-test threats (information provision, framing effects in a new domain).

Want me to do a proper novelty search on any of these, or develop one further?



## Prompt:
18/03/2026, 09:33:40

Ok. Here is my reaction: 

1) Not possible, because the AI-CBT app is only available in the US. Also, fielding mental health stuff to random people on instagram is too dangerous ethically and could very much create backlash. 

2) Really not sure. This could very easily fall into a paper with the punchline "people are confused". This is especially unsurprising in career search, as students have very little information and are super uncertain. 

3) I hate list experiments. I think we could show a gap between stated and real preferences somehow (private vs. public), but not sure this is it. 

4) You saw the objection. This is simple information provision. But the qualitative causal models could lead somewhere. Not sure. Do not want to write a simple descriptive paper about: what are causal models of career choice.

5) yeah. This seems dangerously close to just framing effects. 

Ok final verdict: Nothing of this is it. Please come up with more and TRULY surprise me!!!



## Response:
18/03/2026, 09:34:41

Alright. Let me throw away the conventional playbook and think harder. The constraint is: ~1,000 business econ students, unpaid, recruited via Instagram/career coaching. The sample itself is the asset — what can you do with *this specific population* that you couldn't do elsewhere?

Let me push into weirder territory.

---

**Idea 6: The market for excuses — eliciting WTP for plausible deniability in career decisions**

Here's the setup. Students face a real career-relevant decision in the survey (e.g., choosing between a high-paying but socially undesirable option and a lower-paying but socially approved one — think tobacco industry vs. NGO, or arms manufacturer vs. sustainability startup). Before choosing, you randomize whether they receive an *excuse* — a legitimate-sounding justification for choosing the lucrative-but-stigmatized option ("you can donate the salary difference," "you'll gain skills transferable to impact work," "industry insiders change companies from within").

The twist: in a second stage, you *sell* excuses. You give people a budget and let them purchase justifications before making their choice. The punchline is not that excuses change behavior (that's framing effects). The punchline is that **excuses have a market price, and that price reveals the shadow cost of social norms on labor allocation.** You can estimate how much money the labor market leaves on the table because people lack socially acceptable narratives for taking high-paying jobs in stigmatized industries.

This is economics: there's a price, a market distortion (socially-driven misallocation of talent away from high-productivity firms), and a novel mechanism (the excuse market). It connects to Bursztyn's work on social image but the *mechanism design* angle (pricing excuses, revealing the norm tax) is new. It's not "social pressure exists" — it's "here is the equilibrium price of a norm-compatible narrative, and it predicts real allocation."

*Kill test:* Is it just Bursztyn in a new domain? The excuse-pricing mechanism is genuinely new — nobody has created a market for justifications and used WTP to back out the norm tax. Is it surprising? I think "excuses have a stable, elicitable market price" is not obvious. Will it be dated? Social norms on career choice aren't going away. Is it psychology? The welfare/misallocation angle makes it economics. **Risk: implementing the "excuse market" cleanly in a survey is tricky design work.**

---

**Idea 7: Endogenous preference formation through commitment — do career plans create the preferences that justify them?**

Students report career preferences, then are randomized into publicly committing to a career direction (e.g., writing a LinkedIn-style career statement that they're told will be shown to peers/coaches) vs. keeping their plans private. Four weeks later, you re-survey. The hypothesis: **public commitment doesn't just lock in behavior — it retroactively manufactures the preferences that rationalize the commitment.** People who publicly committed to consulting now *genuinely prefer* consulting more, not because they learned anything, but because cognitive dissonance reshapes preferences to match commitments.

The punchline: Career preferences are *downstream* of career commitments, not upstream. This reverses the standard economic model (preferences → choices) and shows choices → preferences. The welfare consequence is severe: if the labor market sorting process itself distorts the preferences that are supposed to guide it, then revealed preference is fundamentally compromised *specifically in the domain where it matters most* (high-stakes, identity-relevant, public career decisions).

This is Akerlof-and-Dickens (1982) cognitive dissonance in economics, but with a novel empirical design and a domain where the welfare stakes are huge. The economics angle: if commitment creates preferences, then early career lock-in (accepting an offer, announcing your plans) has a *preference externality* on your future self. This connects directly to your Bernheim-Rangel interest — which self's preferences count when commitment manufactures preferences?

*Kill test:* Is it psychology? The preference-externality-on-future-self framing and the welfare analysis make it economics. Is it surprising? "Commitment changes preferences" is known conceptually (Festinger), but "the magnitude is large enough to invalidate revealed preference in career markets" would be genuinely surprising to an econ seminar. Is it just Akerlof-Dickens in a new domain? The *empirical design* (randomized commitment + preference re-elicitation + welfare adjudication) is new, and the connection to Bernheim-Rangel on which preferences are welfare-relevant adds theoretical novelty. **Risk: the 4-week follow-up with unpaid students might have brutal attrition.**

---

**Idea 8: Strategic ignorance in career search — people pay to NOT learn about careers they suspect they'd like**

You offer students a menu of free career information modules (e.g., "what it's like to work in investment banking," "day in the life of a data scientist," etc.). Crucially, you also offer them the option to *avoid* specific information — and you measure which information they actively refuse. The prediction from standard economics: information is always (weakly) valuable, so nobody should refuse it. The behavioral prediction: people will strategically avoid information about careers that might *challenge their current plan*, especially when they've already made partial commitments.

The punchline: **People engage in strategic ignorance about career options — and this ignorance is not random, it's systematically directed at the highest-value alternatives.** The careers people refuse to learn about are precisely the ones they'd be most likely to switch to if informed. This creates a self-reinforcing misallocation: the people most mismatched in their current trajectory are the ones most motivated to avoid learning this.

The economics: this is an information-avoidance version of the Stigler job search model. In Stigler, search costs are exogenous. Here, search costs are *endogenous and strategic* — people manufacture their own information frictions. The welfare loss is computable: you compare the career-preference fit of "informed" vs. "strategically ignorant" students.

This connects to the Golman-Hagmann-Loewenstein (QJE 2017) information avoidance literature but takes it to a domain (labor market sorting) where the welfare stakes and equilibrium implications are much larger. And your sample is perfect — students at a career decision point with real stakes.

*Kill test:* Is it information provision? No — it's about information *refusal*. Is it surprising? "People avoid career-relevant information about their best alternatives" is not obvious — the standard model says they shouldn't, and many people would predict curiosity should dominate. Is it transient? Strategic ignorance is structural, not tech-dependent. Is it economics? The Stigler-search-with-endogenous-costs framing and welfare computation make it firmly economics. **Risk: you need to construct the information menus carefully so that "refusal" is meaningful and not just "I'm busy."**

---

**Idea 9: Revealed preference for conformity in career choice — a coordination game with real stakes**

This one exploits the Instagram/social recruitment channel. You create a "career matching game" where students simultaneously choose a career sector, and their payoff (e.g., entry into a career coaching raffle, or a personalized report quality) depends on *how many others chose the same sector.* This is a pure coordination game, and the equilibria reveal which careers serve as focal points.

In one treatment, you show the social distribution of choices in real time. In the other, you don't. The punchline: **career choice is a coordination game, and the coordination motive alone — independent of any preference or aptitude — concentrates talent in focal-point careers** (consulting, investment banking). The "prestige" of these careers isn't about their intrinsic quality; it's that they're Schelling points in a coordination problem.

The economics: this reframes career clustering not as a preference story or an information story, but as a *strategic coordination* story. The welfare implication is that coordination failures in labor markets can be solved by changing focal points, not by changing preferences or information. This is pure market design thinking applied to career markets.

*Kill test:* Is it surprising? Reframing career prestige as a coordination equilibrium rather than a preference is a conceptual shift that I think would surprise a seminar. Is it psychology? It's game theory and market design. Is it transient? Coordination problems are structural. **Risk: the artificial coordination game might feel too "lab-like" to generalize. You'd need to argue the game captures something real about career markets.**

---

My honest ranking:

**Idea 7 (commitment manufactures preferences)** is the most conceptually sharp. It has one clean punchline, reverses a fundamental economic assumption, connects to Bernheim-Rangel, and the design is elegant. The attrition risk is the main concern.

**Idea 6 (market for excuses)** is the most creative mechanism design. Pricing excuses is genuinely novel and the norm-tax computation is economics. Design complexity is the risk.

**Idea 8 (strategic career ignorance)** is the cleanest empirically — one main outcome (information refusal patterns), pre-registerable, fits the sample perfectly, and the Stigler-with-endogenous-costs framing is strong.

**Idea 9 (coordination game)** is the biggest theoretical swing but hardest to execute convincingly.

Any of these land?



## Prompt:
18/03/2026, 09:40:33

Here are my reactions: 

6) I kind of like this because it addresses shame. I mean super tricky to implement. But let us keep this in the back of our midns

7) yeah, a bit worried about attrition

8) Kind of nice. Not sure. 

9) Yes, this feels very lab-like. 


Ok, let us develop further. For some reason, it keeps staying in my head that I like these psych9ology interviews where people are asked to reflect on their values. We know that this can be a powerful guidance tool which leads to long lasting satisfaction (I am thinking of accceptance and commitment therapy, correct me if this assumption is wrong). WHy I am intriduied by this is that the scalable application of such interviews has not been possible before AI, so this presents a real opportunity. However, this is not economcis and no experiment with far reaching consequences so far. We just have: There is a nice tool that (presumably) will have very high satisfaction among users and provide clear prefeerence discovery. Please brainstorm and think really hard if we could turn something like this into a top-5 paper, what the experiment would need to be etc. Give me ideas!



## Response:
18/03/2026, 09:41:31

OK, let me sit with this. You're right that ACT-style values clarification (identifying core values, committed action aligned with them) has strong evidence for lasting behavioral change — it's one of the more robust findings in clinical psych. And you're right that AI makes it scalable for the first time. But "nice tool, people like it" is not a paper. So the question is: what *economic* question does this tool allow you to answer that you couldn't answer before?

Let me think about what the tool actually *does* at a fundamental level. A values clarification interview doesn't give people new information. It helps them access preferences they already have but can't articulate or act on. In economics language: **it reduces the gap between true preferences and decision-relevant preferences.** That's not information provision — it's preference *extraction*.

This is where it gets interesting.

---

**Idea 10: AI values clarification as a revealed-preference correction device — and the welfare test**

Here's the core insight. Economics assumes revealed preference: your choices reflect your preferences. But there's a known problem — in complex, high-stakes, infrequent decisions (career choice being the canonical example), people often can't access their own preferences well enough for their choices to reveal them. They rely on heuristics, social defaults, and salient attributes (salary, prestige) because their *actual* values (autonomy, creativity, geographic flexibility, work-life integration) are hard to introspect on and even harder to trade off.

An AI values clarification interview is, in economic terms, a **preference-extraction technology**. It doesn't change preferences. It makes latent preferences decision-relevant. This is fundamentally different from information provision (which changes beliefs) and from nudging (which changes choice architecture). It's a third category: *preference clarification*.

Now here's the experiment. You randomize students into three arms:

1. **Control**: Standard career preference survey (rank sectors, rate attributes, etc.)
2. **AI values interview**: A 15-minute AI-guided ACT-style values clarification conversation, then the same career preference survey
3. **Structured reflection placebo**: Same time spent, but doing structured career research (reading job descriptions, salary data), then the same survey

The first-stage outcome is preference *precision* — do post-interview preferences predict actual career behavior better than control preferences? You'd need a follow-up (even 3-6 months later: "did you apply to the sectors you said you preferred?") to measure predictive validity.

But here's where it becomes a top-5 paper. **The real outcome is not whether the tool "works." The real outcome is what the tool reveals about the failure of revealed preference in career markets.**

If preferences move substantially after a 15-minute conversation — and if people *endorse* the movement (your Bernheim-Rangel adjudication idea) — then you've demonstrated that pre-clarification career choices were systematically misaligned with people's own values. That's not "people are confused" — that's **a quantifiable market failure in labor allocation caused by preference inaccessibility.** And unlike information asymmetry or externalities, this market failure has no standard fix in the economics toolkit. You've identified a new class of market failure.

*Kill test:* "People are confused" punchline? No — the punchline is "there exists a quantifiable preference-access friction in labor markets, it's large, and a specific technology eliminates it." That's structural. "Just psychology"? The *tool* is psychology. The *finding* is about labor market allocation efficiency. "Obvious"? I don't think "a 15-minute conversation moves career preferences by X% and the movement is welfare-improving by people's own assessment" has an obvious sign or magnitude. "Information provision"? Explicitly not — the placebo arm (structured career research) controls for information, isolating the preference-extraction channel.

**But I share your concern: this alone might not be enough for a top-5.** So let me push further on what would make it sufficient.

---

**Idea 10a: The preference-extraction technology meets a matching market**

The paper becomes top-5 material if you can show *allocative consequences*, not just individual preference movement. Here's how.

You run the values clarification intervention. Then you create a **real matching market** within the survey. Students are matched to mock internship interviews, career coaching slots, or mentorship sessions based on their stated preferences. The matching is incentive-compatible (they actually get whatever they're matched to through pumpkincareers). Now you can ask: **does preference clarification improve match quality?**

You measure match quality by: (a) ex-post satisfaction with the match, (b) whether they actually show up, (c) whether their stated preferences predict their satisfaction better post-clarification.

The punchline becomes: **Preference inaccessibility causes systematic mismatch in markets, and a scalable preference-extraction technology improves allocative efficiency by X%.** This is a market design paper. The values interview isn't a therapy tool — it's *market infrastructure* that solves a novel friction.

This frames the contribution as: economists have identified information frictions, search frictions, and transaction costs as sources of market failure. Here's a new one — *preference-access frictions* — and here's both the evidence that it matters and a technology that solves it.

---

**Idea 10b: Preference extraction reveals preference *structure* — testing economic models of career choice**

Alternative angle. You use the AI values interview not just as a treatment but as a *measurement device*. The interview transcripts give you rich data on how people actually structure their career preferences — what dimensions they mention spontaneously, how they trade off attributes, where they get stuck.

You compare the preference structure people reveal in the interview to the preference structure economists *assume* in standard career choice models (usually: wage, sector, risk). The punchline: **the dimensions people actually optimize over are radically different from what economic models assume, and this mismatch explains specific, predictable patterns of career misallocation.**

For example, maybe people care enormously about "narrative coherence" (can I tell a good story about my career?) — a dimension that doesn't appear in any economic model but turns out to predict career choice better than wage. Or "reversibility" (can I change my mind?) dominates in ways standard option-value models don't capture.

This is methodological: AI-powered interviews as a new *elicitation technology* for economics, revealing preference dimensions that surveys miss. It's in the spirit of your interest in tacit knowledge — the interview surfaces the implicit causal models and values people act on but can't report in a questionnaire.

*Kill test:* "Descriptive"? Yes, partially — but if you show the newly-discovered dimensions *predict behavior better than standard models*, that's a contribution to economic theory. "Psychology"? The measurement is psychological. The implication — that career choice models are misspecified because they miss key dimensions — is economics.

---

**Idea 10c: The boldest version — preference clarification as Pareto improvement in a real market**

This requires pumpkincareers to cooperate, but: you run the AI values interview for half the students in the coaching program. Then pumpkincareers matches all students to coaching sessions, mentors, or company contacts *based on their stated preferences*. After the match, you measure satisfaction, follow-through, and eventual career outcomes.

If the values-clarified students get better matches (higher satisfaction, higher follow-through) *without any cost to the non-clarified students*, you've demonstrated a **Pareto improvement from preference extraction in a real matching market.** That's the cleanest possible welfare result.

And the mechanism is not information, not nudging, not incentive design — it's a new category entirely. You could even frame it as analogous to the Roth-Shapley matching literature: stable matching assumes preferences are known, but what if they're not? Preference extraction is the missing step *before* the matching algorithm runs.

---

My honest assessment: the core insight — preference-access frictions as a new class of market failure, with AI values interviews as the technology that reveals and solves them — is strong enough for a top-5 if executed well. The question is which version gives you the cleanest identification and most surprising result. 10a (matching market) is the most ambitious. 10b (preference structure) is the most feasible with your current setup. 10c requires pumpkincareers infrastructure but is the most compelling welfare result.

What's your reaction? Does the "preference-access friction" framing resonate, or does it feel like dressing up psychology in economics language?



## Prompt:
18/03/2026, 09:48:16

10) I really think that the values interview might move preferences. However, this seems too obvious: Value reflections is something people often do not do - they rely on heuristics. So it risks being an interesting descriptive finding, with a so what? feeling. It would be nice if everyone reflects, but they dont. So I think we could only make this really good with a long term follow-up, which is too risky for me (attrition). 

10a) Could work. I cannot do the real interviews for everyone. But ok, it could work
10b)  Yes, this is measurement. 
10c) I do not get what the advantage over 10a) is. And it does not work with pumpkincareers. 

Let us do two things in parallell: Develop this further and start with a new observation: AI starts replacing a lot of stuff business students could do. Can we do something with this?



## Response:
18/03/2026, 09:49:29

Good. Let me work both threads.

---

## Thread A: Developing the values interview idea

Let me take your objections seriously. The "so what" problem is real — "people should reflect more but don't" is a finding without a mechanism or a market consequence. And you can't rely on long-term follow-up. So the paper needs to generate its punchline *within the survey itself*.

Here's where I want to push. The interesting question isn't "does reflection move preferences" (probably yes, boringly). The interesting question is: **what does the *pattern* of preference movement tell us about how career markets fail?**

Concretely: if you run the values interview and then compare pre- vs. post-reflection career preferences, the *direction* of movement is informative. If everyone moves toward the same careers post-reflection, that's one story (those careers are genuinely better, people just didn't know). If people *disperse* — moving away from focal-point careers in idiosyncratic directions — that's a much more interesting story: **social defaults compress the preference distribution, and reflection decompresses it.** The implied market failure is that without reflection, too many people cluster in the same careers, creating excess competition in some sectors and undersupply in others. That's allocative inefficiency caused by preference compression.

The outcome you measure isn't "did preferences move" but "did the *variance* of career preferences increase." And you measure it within-session, no follow-up needed. The placebo arm (structured career research/information) is critical: if information also disperses preferences, you've got nothing. But if information *doesn't* disperse (maybe it even compresses, because everyone learns the same salary rankings), while values clarification does, then you've cleanly isolated the preference-access channel.

The punchline: **Labor markets exhibit excess career concentration not because of information failures or coordination problems, but because people default to socially salient options when they can't access their own preferences. A scalable preference-extraction technology decompresses the distribution, reducing excess competition and improving allocative efficiency.**

This also gives you a nice structural estimation possibility: using the pre/post preference distributions, you can compute the welfare cost of preference compression in career markets. That's a number, not a description.

Now combine this with a version of 10a. You don't need real interviews for everyone. You need a *consequential choice* that depends on stated preferences. Through pumpkincareers, could you offer students a menu of, say, 6 different career workshop sessions or networking events (one for consulting, one for tech, one for startups, one for public sector, etc.), and they pick one? Slots are limited, so choices are real. Then the question becomes: does values clarification change which workshop people choose, and does it decompress the distribution of choices across workshops?

If in the control group 60% choose consulting and in the treatment group it's 35%, with the freed-up mass spreading across other options — that's a real allocative result within the survey.

**Does the "preference compression → decompression" framing change your assessment? It shifts the punchline from "reflection is good" to "career markets are inefficiently concentrated for a novel reason."**

---

## Thread B: AI replacing business students' skills

Let me think about what's actually happening here. AI is eating the entry-level knowledge work stack: financial modeling, market research, slide-making, memo-writing, basic data analysis. These are precisely the skills that business economics students are investing in. So there's a human capital story: **students are making human capital investments that are depreciating faster than they realize.**

But "AI makes some skills obsolete" is not a paper — it's a newspaper headline. Where's the economics?

---

**Idea 11: Beliefs about skill depreciation — and the human capital investment distortion**

You elicit students' beliefs about which of their skills will be AI-substituted in 5 years. Then you show them expert/market forecasts (or AI demonstrations). The standard information experiment would just measure updating. Boring, you'd reject it.

So don't do that. Instead, here's the twist: **measure whether beliefs about AI substitution change human capital *investment* decisions right now, within the survey.** You offer students a choice between two real micro-credential workshops or learning modules through pumpkincareers: one in an AI-substitutable skill (e.g., financial modeling in Excel) and one in an AI-complementary skill (e.g., client relationship management, negotiation). This choice is consequential.

The randomization isn't information provision — it's *salience*. In one arm, before choosing, students do a brief task where they compete against an AI on a business skill (and lose). In the other arm, they do a task where they collaborate with an AI (and see the AI can't do the human part). The control just chooses.

The punchline: **Experiencing AI substitution shifts human capital investment toward complements, but experiencing AI collaboration does not.** If that's the finding, it tells you something structural about how labor markets will adjust: adjustment happens through threat salience, not through rational updating. This means the transition will be *lumpy* — mass adjustment when people personally experience substitution, not gradual reallocation as information trickles in.

*Kill test:* Is it surprising? I think the asymmetry (substitution experience shifts investment, collaboration experience doesn't) would be surprising. Is it transient? The *specific* skills change, but the mechanism (salience-driven human capital adjustment) is structural. Is it economics? Human capital investment with endogenous belief updating — yes. **Risk: "experiencing something makes it salient" might be too close to availability heuristic, which is just psych.**

---

**Idea 12: The productivity illusion — AI inflates perceived competence and distorts labor market signaling**

Here's something potentially sharper. You give students a business task (write a market analysis, build a financial projection, draft a strategy memo). Half get AI assistance, half don't. Then — crucially — you elicit their *self-assessed competence* at the task, and their *confidence that they could do this in a job interview or on the job without AI.*

The hypothesis: AI assistance inflates self-assessed competence. People who used AI to produce a great memo now believe *they* can produce great memos. This is not about the output quality — it's about the **signal distortion in the labor market.** If AI inflates perceived competence, then:

1. Students apply for jobs they can't actually do (without AI)
2. Employers can't distinguish AI-augmented competence from genuine competence
3. The resulting matches are systematically worse than in a world with accurate self-assessment

This is a **lemons problem created by AI.** Not in the product market (Akerlof's original domain) but in the *labor market*. AI is the technology that makes lemons look like peaches — to themselves and to employers. The unraveling logic follows: if employers suspect AI-inflation, they discount all signals, which hurts genuinely competent people. Classic adverse selection.

The design within your survey: after the task, you offer students the option to "certify" their ability by doing a follow-up task *without AI*, with the result shown to potential employers/coaches through pumpkincareers. The take-up of certification reveals who knows they're inflated vs. who genuinely believes they're competent. This gives you a *separation mechanism* and lets you test whether the market can self-correct.

*Kill test:* Is it just "Akerlof in a new domain"? The mechanism by which lemons are *created* (AI inflates self-assessment) is novel — in Akerlof, quality is fixed and the problem is information. Here, the *perception* of quality is endogenously distorted by a technology. Is it surprising? "AI makes people overconfident" might seem obvious, but "this overconfidence creates an adverse selection problem in labor markets with a specific unraveling structure" is a theoretical result, not a psychological observation. Is it transient? Even as AI improves, the gap between "I produced this with AI" and "I can do this" persists — arguably it *widens*. Is it economics? Adverse selection, signaling, market design — yes. **This feels like it has a single non-obvious conceptual punchline: AI creates lemons in labor markets.**

---

**Idea 13: The skill-bundling problem — AI unbundles jobs and students don't know how to rebundle**

More speculative. Business jobs are *bundles* of tasks (consulting = analysis + slides + client management + travel tolerance + long hours). AI substitutes some tasks in the bundle but not others. Students currently invest in the *bundle* because that's how jobs are packaged. But if AI unbundles the task structure, the optimal human capital portfolio changes — and nobody knows what the new bundles should be.

You could measure this by giving students the full task decomposition of their target careers (10-15 tasks per career, with AI-substitutability ratings for each) and seeing whether this changes their career preferences or their skill investment choices. The interesting finding would be if students *can't actually rebundle* — they don't know how to construct a human capital portfolio from individual tasks rather than from job titles.

*Kill test:* This feels more like a framework contribution than an experiment. It might be too conceptual / too many moving parts to be a clean paper. "Students don't know how to think about tasks vs. jobs" is also potentially obvious. **Verdict: interesting as a framing device but probably not the main paper.**

---

My ranking on Thread B:

**Idea 12 (AI creates lemons in labor markets)** is the strongest. It has the Akerlof-style single punchline you favor: a technology that was supposed to increase productivity instead *distorts the signal structure of the market*, creating adverse selection where there was none before. The certification mechanism gives you a within-survey test of market self-correction. And the sample is perfect — business students are exactly the population experiencing this right now.

**Idea 11** is decent but risks the "salience is just availability heuristic" objection.

**Idea 13** is a nice framework but not a clean experiment.

---

So across both threads, the two strongest candidates are:

1. **Thread A: Preference compression/decompression** — values clarification reveals that career markets are inefficiently concentrated because people can't access their own preferences. Novel friction, measurable within-session.

2. **Thread B: AI-created lemons** — AI inflates self-assessed competence, creating adverse selection in labor markets. Novel mechanism for a classic market failure, testable with certification.

Where do you want to dig deeper?



## Prompt:
18/03/2026, 09:54:21

Thread A: With this pitch, inequality keeps popping up in my head. And, I would need to show it is really SOCIAL norms that lead to supression. But I start loving this - it is a Bursztyn style social norms affect market allocations survey. Let us keep refining. 

Idea 11: Yeah, do not like this. People already know how AI currently works


But I am more convinced by thread A, because I could actually field it and provide some value to people. So we could get a larger sample size here. 

Let us develop this into a real top 5.



## Response:
18/03/2026, 09:57:00

````plaintext
Thought process: Architected rigorous design separating internalized versus expressed preference compression mechanisms.

Nico is excited about Thread A - the preference compression/decompression idea via values clarification interviews. He's making two key observations:

1. Inequality keeps popping up - there might be a heterogeneity angle, but he hates heterogeneity papers. So I need to think about how inequality enters as a *main effect* or structural prediction, not as "it works for subgroup X but not Y."

2. He wants to show it's really SOCIAL NORMS that lead to suppression - this is the Bursztyn connection. The mechanism isn't just "people don't reflect enough" but "social norms actively compress preferences away from what people actually want."
   
The values interview also gives me a real benefit to offer participants, which helps justify a larger sample size. For this to be a top-tier paper, the core insight needs to be sharp and unexpected - something like social norms don't just constrain what people do, they constrain what people can even think about wanting. The design then needs a clean treatment that isolates social norm pressure as the driving mechanism.

I'm working through the design logic: I could cross-randomize the values interview against public versus private preference reporting. If the interview shifts preferences more in private settings, that suggests social norms suppress expression. But the stronger finding would be if the interview moves preferences even when reporting is private - meaning social norms have been so deeply internalized that people can't access their true preferences even when no one's watching. That's different from Bursztyn's work, which shows norms affect what people say; this would show norms affect what people actually want to experience.

The challenge is distinguishing whether the values interview is revealing suppressed preferences or just clearing up confusion.

I need to build in variation in social norm pressure itself. One approach: randomize whether students see peer choice information before the values interview - like showing them that 70% of peers chose consulting - then see if the interview's effect is stronger for those primed with norm information. But actually, a cleaner design would be to have everyone do the values interview, then vary whether the preference question is public or private. The gap between public and private responses shows how much remaining compression is strategic versus internalized, while the shift from before to after the interview reveals how much was internalized norm compression that the interview undid.

I'm thinking a 2x2 design could really sharpen this... - Values interview vs. placebo (career info)
- Public vs. private preference reporting

The key predictions:
1. In private reporting: values interview disperses preferences (decompresses). This shows internalized norm compression.
2. In public reporting: values interview has LESS effect (social image concerns re-compress). 
3. The gap between private-interview and public-interview reveals the *social norm tax* on preference expression.
4. The gap between private-control and private-interview reveals the *internalized norm tax* on preference access.

This gives you TWO types of norm-based market failure: (a) people can't express their true preferences publicly (Bursztyn already showed variants of this), and (b) people can't even ACCESS their true preferences because norms have been internalized (this is NEW).

The punchline: social norms distort labor market allocation through two channels — preference expression and preference access. The values interview separates them, and the internalized channel might actually be larger than the expression channel, so even perfect privacy wouldn't fix career market distortions.

Now I'm thinking about how inequality fits in without just becoming a heterogeneity story. What if the main finding itself IS about inequality — that students from different backgrounds have similar latent preferences once the values interview decompresses them, but very different expressed preferences beforehand, with the gap driven by differential norm pressure? That would mean social norms are actually a mechanism generating career inequality across groups. (social norms channel) and larger for students whose social environment has stronger career norms (identifying the source of compression).

**Allocative consequence:** You compute a misallocation index — how much excess concentration exists pre-interview, and how much the interview resolves. You show this maps to real choices (workshop selection through pumpkincareers).

**Welfare statement:** The welfare cost of preference compression in career markets is X, and it's caused by social norms, not information deficits.

Now I need to think through what would actually convince Nico to develop this into a full paper. I should nail down the exact experimental design, figure out how to identify the causal mechanism, clarify what the main outcome variable is, and articulate what would make this publishable at a top venue. I also need to do a quick check on whether this is genuinely novel or if there's existing work that covers similar ground.
````

OK. Let me think about this properly — design, identification, punchline, and what separates this from a good paper and a top-5 paper.

````plaintext
Web Search: social norms career choice experimental economics
````

> **Women's career choices, social norms and child care policies - ScienceDirect** [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0047272718301944)
>
> **Journal of Economic Behavior & Organization | Social Norms & Behavior Change in Economics | ScienceDirect.com by Elsevier** [sciencedirect.com](https://www.sciencedirect.com/special-issue/10QZ6D4RXKX)
>
> **The influence of social norms on economic decision-making** [alliedacademies.org](https://www.alliedacademies.org/articles/the-influence-of-social-norms-on-economic-decisionmaking-29664.html)
>
> **Social Norms (Behavioural Economics) | Reference Library | Economics | tutor2u** [tutor2u.net](https://www.tutor2u.net/economics/reference/behavioural-economics-social-norms)
>
> **Experimental methods: Eliciting and measuring social norms - ScienceDirect** [sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0167268125003063)
>
> **Personal norms — and not only social norms — shape economic behavior - ScienceDirect** [sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0047272724001919)
>
> **Social influence in career choice: Evidence from a randomized field experiment on entrepreneurial mentorship - ScienceDirect** [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0048733317300100)
>
> **Social norms in survey experiments: Personal beliefs versus normative expectations - ScienceDirect** [sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0167268125004093)
>
> **Social influence in career choice: Evidence from a randomized field experiment on entrepreneurial mentorship** [repec.org](https://ideas.repec.org/a/eee/respol/v46y2017i3p636-650.html)
>
> **Measuring social norms and preferences using experimental ...** [dklevine.com](http://www.dklevine.com/archive/refs4506439000000000501.pdf)
>
````plaintext
Thought process

Let me search more specifically for the closest threats - Bursztyn's work on social norms and career choice, and any work on preference compression/decompression.
````

````plaintext
Web Search: Bursztyn social norms career aspirations labor market allocation experiment
````

> **Experimental Evidence on the Labor Market Aspirations of ...** [worldbank.org](https://thedocs.worldbank.org/en/doc/9956d1747b25078daae65b260f3fbfe1-0280032023/original/Information-and-Social-Norms-Experimental-Evidence-on-the-Labor-Market-Aspirations-of-Saudi-Women.pdf)
>
> **A Family of Norms: Related Beliefs about Female Labor Force Participation ∗** [pitt.edu](https://sites.pitt.edu/~swwang/papers/SA_Social_Norms.pdf)
>
> **Misperceived Social Norms: Women Working Outside the Home in Saudi Arabia** [researchgate.net](https://www.researchgate.net/publication/345893128_Misperceived_Social_Norms_Women_Working_Outside_the_Home_in_Saudi_Arabia)
>
> **Social Norms and Economic Behavior: Gender Norms, Perceptions, and their Impact** [worldbank.org](https://thedocs.worldbank.org/en/doc/d390fb024090f35d35dfbe0f4e49498c-0310022023/original/Social-Norms-and-Economic-Behavior-Bursztyn.pdf)
>
> **Experimental Evidence on the Labor Market Aspirations of ...** [nber.org](https://www.nber.org/system/files/working_papers/w26693/w26693.pdf)
>
> **Misperceived Social Norms: Women Working Outside the ...** [uchicago.edu](https://home.uchicago.edu/bursztyn/Misperceived_Norms.pdf)
>
> **Data and Code for: Misperceived Social Norms: Women Working Outside the Home in Saudi Arabia** [openicpsr.org](https://www.openicpsr.org/openicpsr/project/119404/version/V1/view)
>
> **Do changing gender norms counteract the effects of information on the aspirations of Saudi women? | VoxDev** [voxdev.org](https://voxdev.org/topic/labour-markets/do-changing-gender-norms-counteract-effects-information-aspirations-saudi)
>
> **Reframing Active Labor Market Policy: A Field Experiment ...** [github.io](https://lukaslehner.github.io/assets/Reframing_ALMP.pdf)
>
> **Moving the needle on social norms** [worldbank.org](https://blogs.worldbank.org/en/impactevaluations/moving-needle-social-norms)
>
````plaintext
Thought process

Good. Now let me search for the most direct threat - anything on preference clarity/reflection and career choice, or AI-guided values clarification.
````

````plaintext
Web Search: preference discovery reflection career choice misallocation welfare economics experiment
````

> **Preference discovery | Experimental Economics | Cambridge Core** [cambridge.org](https://www.cambridge.org/core/journals/experimental-economics/article/abs/preference-discovery/8FDA8EAAC91723A8593F24E552680B99)
>
> **DISCUSSION PAPER SERIES IZA DP No. 16200 Ernst Fehr Gary Charness** [iza.org](https://docs.iza.org/dp16200.pdf)
>
> **Prestige Seeking in College Application and Major Choice Yeon-Koo Che** [columbia.edu](https://econ.columbia.edu/wp-content/uploads/sites/32/2019/05/Prestige.pdf)
>
> **Revealed Preferences - an overview | ScienceDirect Topics** [sciencedirect.com](https://www.sciencedirect.com/topics/economics-econometrics-and-finance/revealed-preferences)
>
> **Discrete Choice Experiment - The Decision Lab** [thedecisionlab.com](https://thedecisionlab.com/reference-guide/statistics/discrete-choice-experiment)
>
> **Behavioral Welfare Economics - PMC** [nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC7264445/)
>
> **Using economic experiments to assess the validity of stated preference contingent behavior responses - ScienceDirect** [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0095069622000377)
>
> **Choosing the Future: Economic Preferences for Higher Education Using Discrete Choice Experiment Method | Research in Higher Education | Springer Nature Link** [springer.com](https://link.springer.com/article/10.1007/s11162-019-09572-w)
>
> **Quality heterogeneity and misallocation: The welfare benefits of raising your standards - ScienceDirect** [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0022199621001240)
>
> **On the relation between preference and choice - ScienceDirect** [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S1053535702001397)
>
````plaintext
Thought process

Good. Let me now search for the most direct threat - Bursztyn acting wife paper and the prestige-seeking paper, which are closest to what we're developing.
````

````plaintext
Web Search: Bursztyn "acting wife" social norms suppress career ambition public private
````

> **'Acting Wife': Marriage Market Incentives and Labor Market Investments | NBER** [nber.org](https://www.nber.org/papers/w23043)
>
> **'Acting Wife': Marriage Market Incentives and Labor Market Investments by Leonardo Bursztyn, Thomas Fujiwara, Amanda Pallais :: SSRN** [ssrn.com](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2900048)
>
> **'Acting Wife': Marriage Market Incentives and Labor ...** [researchgate.net](https://www.researchgate.net/publication/320770197_'Acting_Wife'_Marriage_Market_Incentives_and_Labor_Market_Investments)
>
> **'Acting Wife': Marriage Market Incentives and Labor ...** [harvard.edu](https://scholar.harvard.edu/files/pallais/files/acting_wife.pdf)
>
> **'Acting Wife': Marriage Market Incentives and Labor Market Investments - American Economic Association** [aeaweb.org](https://www.aeaweb.org/articles?id=10.1257/aer.20170029)
>
> **'Acting Wife': Marriage Market Incentives and Labor ...** [uchicago.edu](https://home.uchicago.edu/bursztyn/Bursztyn_Fujiwara_Pallais_ActingWife.pdf)
>
> **'Acting Wife': Marriage Market Incentives and Labor Market Investments** [repec.org](https://ideas.repec.org/a/aea/aecrev/v107y2017i11p3288-3319.html)
>
> **Replication data for: 'Acting Wife': Marriage Market Incentives and Labor Market Investments** [openicpsr.org](https://www.openicpsr.org/openicpsr/project/113174/version/V1/view)
>
> **‘Acting Wife’: Marriage Market Incentives and Labor Market Investments∗** [apsanet.org](http://web.apsanet.org/cswp/wp-content/uploads/sites/4/2016/04/Bursztyn_Fujiwara_Pallais_Dec2016.pdf)
>
> **‘Acting wife’: Marriage market incentives and labor market investments - Princeton University** [princeton.edu](https://collaborate.princeton.edu/en/publications/acting-wife-marriage-market-incentives-and-labor-market-investmen)
>
````plaintext
Thought process: Architected experimental design separating preference suppression from internalization mechanisms.

OK, now I have a very clear picture of the closest papers. Let me develop this properly.

The key reference is Bursztyn, Fujiwara, Pallais (2017) "Acting Wife" - which shows that social norms suppress career preferences in public settings. The design: public vs. private elicitation of career preferences. They find single women downplay ambition when peers observe.

Now Nico wants to go beyond this. The "Acting Wife" finding is about EXPRESSION - people have preferences but hide them. What Nico is interested in is whether social norms also affect preference ACCESS - do people not even know what they want because norms have prevented them from reflecting?
 
The values interview is the tool that can separate these two channels. I need a 2x2 design crossing the interview intervention with public versus private settings. This lets me isolate the expression channel—the gap between public and private responses—from the access channel, which would show up as a gap between the interview and control conditions in private settings where expression suppression shouldn't matter.

If the internalized channel is substantial, that suggests social norms don't just suppress what people say but actually constrain what they can think or access about their own preferences. That's a more fundamental market failure than simple preference hiding.

For inequality, I should avoid framing it as just heterogeneous effects. Instead, the core finding could be about how the internalized channel operates as the main mechanism, and then I show that this misallocation concentrates among lower socioeconomic groups—making inequality central to the welfare story rather than a side analysis.

Option B feels right: the paper's punchline is that career sorting inequality across SES groups looks like it stems from real preference differences, but much of it actually comes from differential norm internalization. A preference-extraction technology that bypasses these norms would then reveal the artificial component of that inequality. This frames it as explaining a cause of inequality through mechanism, not just documenting that effects vary by group.

Now I'm thinking through the cleanest framing: social norms compress career preferences through internalization rather than just suppressing their expression, creating measurable welfare losses that a scalable intervention like an AI values interview could address.
````

OK. Let me now put this together properly. The novelty search is informative — the closest paper is Bursztyn, Fujiwara & Pallais (2017) "Acting Wife," which established that social norms suppress *expressed* career preferences when peers observe. The gap in the literature is clear: nobody has shown that social norms suppress *preference access itself* — that people can't even articulate their true career preferences in private because norms have been internalized. And nobody has had a scalable tool to undo that internalization.

Here's the paper.

---

## Working title: "Knowing What You Want: Social Norms, Preference Access, and Career Misallocation"

### The conceptual punchline (one sentence)

Social norms don't just prevent people from *expressing* their career preferences — they prevent people from *knowing* their career preferences, and this internalized preference suppression causes quantifiable misallocation that a scalable AI-based preference-extraction technology can reverse.

### Why this is new

Bursztyn et al. (2017) showed the *expression* channel: single women downplay ambition when peers observe, but report their true preferences in private. That's social image management — people know what they want but hide it. The paper I'm proposing documents a *deeper* channel: even in fully private settings, people's stated career preferences are compressed toward social defaults because they haven't accessed their own latent values. The values interview doesn't give people information or change their incentives — it gives them *access to preferences they already have but couldn't articulate*. This is a new class of friction that sits between Bursztyn's social image costs and standard information frictions.

### The design (2×2)

**Treatment 1: Values interview (AI vs. placebo)**

- **AI values interview** (~15 min): An ACT-inspired AI conversation that asks students to reflect on what matters to them in life, what they'd regret not doing, what their ideal Tuesday looks like in 10 years, etc. Crucially, this conversation *never mentions careers, sectors, or salaries.* It operates purely at the values level. This is important because it means any career preference movement post-interview cannot be attributed to career information or career-specific priming.
- **Placebo**: Same time investment, but doing a structured career information task (reading sector descriptions, salary data, employment statistics). This controls for time, engagement, and career-relevant information.

**Treatment 2: Social norm salience (high vs. low)**

Before stating career preferences, students are randomized into:
- **Norm-salient condition**: "Here is what other students in your program ranked as their top career choices: [show the actual modal distribution, which will be heavily concentrated in consulting/banking/tech]." This makes the social default salient.
- **Neutral condition**: No peer information.

Then *all students* state career preferences privately (ranking sectors, rating attributes, reporting intended career plans). Privacy is important and should be emphasized: "Your responses are completely anonymous and will not be shared with anyone at your university."

### The four cells and what they tell you

| | No norm salience | Norm salience |
|---|---|---|
| **Placebo** | Baseline preferences (cell A) | Norm-compressed preferences (cell B) |
| **Values interview** | Clarified preferences (cell C) | Clarified + norm re-compressed? (cell D) |

The key comparisons:

**C vs. A (private, interview vs. placebo)** = the *internalized suppression* channel. This is the main finding. If preferences move substantially — and specifically, if they *disperse* (variance increases) — after a values conversation that never mentioned careers, then people's baseline career preferences were suppressed by something other than information deficits. Since elicitation is private (no social image concerns), the suppression must be *internalized norms.*

**B vs. A (placebo, norms vs. no norms)** = the *norm reinforcement* effect. Does making social defaults salient compress preferences further? If yes, it confirms the mechanism is social.

**D vs. C (interview, norms vs. no norms)** = *resilience of clarified preferences*. Do post-interview preferences resist social norm pressure? If the interview "immunizes" people against norm compression, that's a powerful result about the durability of preference extraction.

**The variance test**: The main outcome isn't mean preference shift (which direction would you even predict?). It's the *dispersion* of the preference distribution. The prediction: values clarification increases variance (people spread out across careers), while norm salience decreases variance (people cluster in focal-point careers). This is the "preference decompression" finding.

### The welfare measurement (within-session, no follow-up needed)

After stating their career preferences, all students answer the Bernheim-Rangel adjudication question: *"Which set of career preferences better reflects what you truly want — the ones you'd state before or after a conversation about your values?"* (for the interview group) or a hypothetical version for the control.

Additionally, you measure **preference-choice consistency**: you offer all students a real, consequential choice from a menu of career workshops/events provided through pumpkincareers (consulting workshop, startup workshop, public sector workshop, creative industries workshop, etc.). Slots are limited. You compare whether *stated preferences predict actual choices* better in the interview group than in the control. If yes, that's a revealed-preference test of preference clarity — the interview made people's choices more consistent with their stated preferences, which is a *welfare improvement by the Bernheim-Rangel criterion.*

The welfare statement becomes: "In the control group, X% of students make workshop choices inconsistent with their stated preferences. In the interview group, this drops to Y%. The implied misallocation rate in career markets due to preference inaccessibility is [X-Y]%."

### Why inequality enters naturally (without being a heterogeneity result)

You mentioned inequality keeps popping up. Here's how it enters as part of the *main* finding, not as heterogeneity. You measure the *concentration* of career preferences — specifically, the share of students whose top-ranked career is in the modal category (say, consulting/banking). In the control group, this concentration is high. If the values interview reduces it, the *freed-up mass* has to go somewhere. Where does it go?

The paper's inequality angle: **preference compression creates artificial labor supply concentration, which depresses wages in crowded sectors and inflates them in undersupplied ones.** The students who are "freed" by the interview to pursue non-default careers are precisely the ones whose talents were being wasted in the crowded sector. This is a labor supply distortion story, not a heterogeneity story. You can even do a back-of-envelope welfare calculation: given the salary distributions across sectors and the supply shift implied by your treatment effect, how much aggregate surplus is lost by norm-driven career concentration?

And if you want to sharpen the inequality dimension further: the *degree of compression* in the control group likely correlates with the *strength of social norms in students' environments.* First-gen students, students from more traditional backgrounds, students whose social circles are more career-homogeneous — they might show more compression. But this enters as a *mechanism test* (what predicts compression?) not as a heterogeneity result (the effect is larger for group X). The main effect is aggregate compression and its allocative costs.

### What makes this top-5

Three things that separate this from a good field journal paper:

1. **New type of market failure.** Economists know about information frictions, search frictions, transaction costs, and social image costs. This paper identifies *preference-access frictions* — people make suboptimal career choices not because they lack information or face social pressure, but because they literally cannot access their own preferences. That's a conceptual contribution.

2. **The AI angle is structural, not gimmicky.** The AI interview isn't the point — it's the *instrument* that reveals the friction. Before AI, you couldn't run a Socratic values conversation at scale. Now you can. So this paper is also about what new research questions become answerable when AI enables preference extraction at scale. That's a methodological contribution that extends far beyond career choice.

3. **The Bursztyn connection.** The paper explicitly extends the Bursztyn social norms program by showing there's a *second channel* (internalization, not just expression suppression) that's potentially larger. If the private-to-private preference shift (C vs. A) is bigger than the public-to-private shift (Bursztyn's "Acting Wife" finding), you've shown the iceberg under the surface: the part of norm-based career distortion that we've been missing because we only looked at expression.

### Practical feasibility

- **Sample**: ~1,000 business econ students, fits perfectly
- **Value to participants**: The values interview is genuinely useful career guidance. Students recruited via Instagram/pumpkincareers would want this
- **Cost**: The AI interview costs API calls. The workshop choices can be provided through pumpkincareers. No participant payments needed
- **Timeline**: Single-session survey + consequential workshop choice. No follow-up required for the main results (though a follow-up would be a bonus)
- **Pre-registration**: One main outcome (preference distribution variance), one mechanism test (norm salience interaction), one welfare measure (preference-choice consistency). Clean, pre-registerable

### Main risks and how to address them

**"The interview just makes people think more — it's not about norms."** The placebo arm (structured career information/research) controls for general reflection. If the values interview moves preferences but the career research placebo doesn't, the mechanism is specifically about value-level introspection, not just "thinking harder."

**"Private preferences might still reflect social desirability."** True — this is why the norm-salience treatment is important. If making norms *more* salient compresses preferences further (B vs. A), and if the interview *undoes* this compression (D vs. B), you've pinned the mechanism on social norms specifically.

**"The interview might just be demand effects."** The interview never mentions careers. It talks about values, life goals, regrets. Any career preference movement is *downstream* of values clarification, not suggested by the interviewer.

---

What's your reaction? The key question I'd want your gut on: does the "internalized preference suppression as a new channel beyond Bursztyn's expression suppression" framing feel like a genuine conceptual contribution, or does it feel like a smaller increment than I'm claiming?





---
Powered by [Claude Exporter](https://www.ai-chat-exporter.net)