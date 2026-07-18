# Annotation guide — behavioural intent taxonomy

This guide is for the human re-audit of the labelled corpus. It defines
each class with positive/negative examples, boundary cases, overlap
rules, and priority rules. **It does not retroactively change any
existing label** — labels change only through explicit human review via
the annotation queue (`reports/annotation/review_queue.csv`).

The unit of annotation is a single public comment. Label the *behaviour
expressed by the commenter*, not the topic of the comment.

If a comment expresses more than one behaviour, choose the primary
label using the priority rules below and record the second behaviour in
`secondary_label` with `multi_intent_flag = 1`.

## Priority rules (when multiple classes apply)

1. Explicit disengagement (`expectation_decay`) beats everything —
   a stated refusal to buy overrides nostalgia or frustration in the
   same comment.
2. Explicit purchase/play commitment (`high_intent`) beats implicit
   signals (nostalgia, interest).
3. First-time-player signals (`new_customer_interest`) beat nostalgia.
4. Accessibility complaints (`confusion_barrier`) beat generic
   frustration when the obstacle is understanding, not waiting.
5. Coping humour (`content_drought_fatigue`) beats `frustrated_demand`
   when the tone is resigned/comic rather than demanding.
6. `frustrated_demand` beats `nostalgia_reactivation` when the demand is
   explicit.
7. `general_discussion` only when no behavioural signal applies.

---

## high_intent

**Definition:** Explicit stated intent to purchase, pre-order, or play —
including replaying or completing the series ahead of the new release.

- **Positive:** "Pre-ordering the second it's available." / "Buying the
  whole collection on Steam this week."
- **Negative:** "This game will sell millions" (market prediction, not
  personal intent → general_discussion). "I'd buy it if it ever came
  out, lol" (sarcastic coping → content_drought_fatigue).
- **Boundary:** Purchase of an *old* title can still be high_intent (it
  is real purchasing behaviour), but note it in `review_notes`.
- **Overlap:** With nostalgia — "grew up with KH, buying day one" →
  high_intent (priority rule 2), secondary nostalgia_reactivation.

## nostalgia_reactivation

**Definition:** Legacy emotional attachment that is pulling the person
back toward the franchise (re-engagement), without an explicit purchase
commitment.

- **Positive:** "Haven't touched KH since 2010, this trailer made me
  want to replay everything."
- **Negative:** "KH2 was my childhood" with no re-engagement signal →
  general_discussion (nostalgia *without* reactivation).
- **Boundary:** Memory-sharing alone is not reactivation; look for a
  present-tense pull ("makes me want to", "downloading it again").

## new_customer_interest

**Definition:** Signals from someone who has never played and is
considering entering the franchise, or barriers-to-entry framing from a
prospective new player.

- **Positive:** "Never played KH — is 4 a good place to start?"
- **Negative:** Veteran discussing accessibility for others →
  general_discussion.
- **Overlap:** With confusion_barrier — if the confusion is what blocks
  a *prospective* player, prefer new_customer_interest and record
  confusion_barrier as secondary.

## frustrated_demand

**Definition:** Demand that clearly exists but is blocked by lack of
news, trailers, release dates, or updates; the commenter is asking or
demanding.

- **Positive:** "Three years since the announcement and not one word.
  Give us a date."
- **Negative:** "lol we survive on crumbs" (resigned humour →
  content_drought_fatigue). "I've stopped caring" (→ expectation_decay).
- **Boundary vs content_drought_fatigue vs expectation_decay:** demand
  present + demanding tone → frustrated_demand; demand present +
  resigned/coping tone → content_drought_fatigue; demand *withdrawn* →
  expectation_decay.

## content_drought_fatigue

**Definition:** Coping signals during the content drought — memes,
resignation, gallows humour ("copium", "crumbs", "we'll take anything").

- **Positive:** "At this point I'll take a screenshot of a doorknob."
- **Negative:** Direct angry demands → frustrated_demand.

## confusion_barrier

**Definition:** Narrative or structural complexity reducing
accessibility — the person cannot follow the story, the play order, or
which titles are required.

- **Positive:** "What order do I even play these in? The timeline makes
  no sense."
- **Negative vs general_discussion:** Affectionate jokes about the lore
  being weird, from an engaged fan, are general_discussion. Label
  confusion_barrier only when the confusion functions as an obstacle to
  engagement or purchase.

## expectation_decay

**Definition:** Disengagement after prolonged silence — stated loss of
interest, trust, or purchase intent.

- **Positive:** "I've moved on. Not buying it whenever it comes out."
- **Negative:** "Still waiting..." (still engaged → frustrated_demand).

## general_discussion

**Definition:** Engagement with no actionable behavioural signal — lore
theories, music appreciation, reactions, chat.

- **Boundary:** Positive sentiment is *not* actionable by itself. "This
  trailer is beautiful" is general_discussion.
