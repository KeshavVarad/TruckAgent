# Truco — Game Rules

This document defines the rules of the two-player Truco variant implemented in this project. These rules are the source of truth for the game engine and the RL environment.

> The game is commonly spelled *Truco* (sometimes written as *Truc*, *Truck*, or *Truqo*). The Brazilian two-player variant described here is the target of the RL agent.

---

## 1. Players and Objective

- **Players:** Exactly 2.
- **Objective:** Be the first player to reach **12 points**.

## 2. The Deck

A standard 52-card deck with the **8s, 9s, and 10s removed** — leaving **40 cards** in 4 suits (♦ Diamonds, ♠ Spades, ♥ Hearts, ♣ Clubs).

## 3. Card Ranking

Cards rank from **lowest to highest** as follows:

```
4 < 5 < 6 < 7 < Q < J < K < A < 2 < 3 < manilhas
```

Above 3 sit the four **manilhas** (trump cards), which are determined each round (see §4). Manilhas always beat any non-manilha card. Among the four manilhas, suit breaks ties in this order (lowest to highest):

```
♦ Diamonds  <  ♠ Spades  <  ♥ Hearts  <  ♣ Clubs
```

So **♣ of the manilha rank** is the single strongest card in the round.

## 4. Dealing and the Manilha (Trump)

At the start of each round:

1. Each player is dealt **3 cards** face down.
2. One additional card is dealt face up — this card is called the **vira**.
3. The **manilha rank** is the rank **one above the vira's rank** in the order shown in §3.
   - Example: if the vira is a 4, the four 5s are the manilhas.
   - Example: if the vira is a K, the four As are the manilhas.
   - **Wrap-around:** if the vira is a 3 (the highest non-manilha rank), the manilhas wrap to the **4s**.
4. All four cards of the manilha rank act as trumps for that round, ordered by suit (♦ < ♠ < ♥ < ♣).

The vira itself does **not** participate in tricks — it is only used to determine the manilha rank.

## 5. The Round (Hand)

Each round consists of up to **3 tricks**. The round is played for a stake, which starts at **1 point** and may be raised (see §7).

### 5.1 Who leads

- Players **alternate who leads the first trick** of each round.
- After the first trick, **the winner of a trick leads the next trick**.

### 5.2 Playing a trick

On a trick, each player plays exactly one card. The **highest card wins the trick** under the ranking in §3.

### 5.3 Empate (tied trick)

If both players play cards of equal effective rank (e.g., both play a K of a non-manilha suit), the trick is an **empate** — neither player wins it.

- After an empate, **the player who led the empate trick leads the next trick**.

### 5.4 Winning the round

The round ends as soon as one of the following is true:

1. A player wins **2 of the 3 tricks** → that player wins the round.
2. All 3 tricks have been played.

At the end of 3 tricks, if neither player has 2 outright wins, the round is decided by the **first-non-tied-trick rule**:

- If trick wins are tied (including via empates), the round goes to the player who **won the earliest non-tied trick**.
- If **all 3 tricks are empates**, the round goes to the player who **led the first trick**.

The winner of the round receives the current stake in points (1, 3, or 6 — see §7).

## 6. Scoring and End of Match

- Round winner receives the round's current stake.
- Match ends immediately when a player reaches **12 points**. That player wins.

## 7. Raising the Stakes — Truco and Seis

The stake starts at **1 point** each round and can be raised through calls.

### 7.1 When you can call

A player may issue a call **only on their own turn**, **before they have played a card on the current trick**. A call replaces the act of playing for that turn — once the call is resolved, play resumes from where it was.

### 7.2 Truco (raise to 3)

- At a stake of 1, either player (on their turn, before playing) may call **Truco**.
- The opponent must respond:
  - **Accept** → the stake becomes **3**, and play continues. The caller now plays their card.
  - **Fold** → the round ends immediately. The caller wins the round at the **previous stake (1 point)**.
  - **Re-raise to Seis** → see §7.3.

### 7.3 Seis (raise to 6)

A **Seis** call is available in either of these situations:
- As a **counter-raise** in immediate response to a Truco call, or
- As a **later raise** on any subsequent turn after a Truco has been accepted (either player may make this call on their own turn, before playing).

The opponent of the Seis caller must respond:
- **Accept** → the stake becomes **6**, and play continues.
- **Fold** → the round ends immediately. The Seis caller wins the round at the **previous stake (3 points)**.

**Seis is the cap.** No further raises are possible after Seis.

### 7.4 Summary table

| Current stake | Available call | If accepted | If folded (caller wins) |
|--------------:|:--------------:|:-----------:|:-----------------------:|
| 1             | Truco          | 3           | 1                       |
| 3 (accepted Truco) | Seis     | 6           | 3                       |
| 6             | — (capped)     | —           | —                       |

## 8. Turn Order Summary (per round)

1. Determine who leads trick 1 (alternates each round).
2. **Trick 1:** Leader's turn → opponent's turn.
3. **Trick 2:** Winner of trick 1 leads (or leader of trick 1 if trick 1 was empate) → opponent.
4. **Trick 3:** Same rule as trick 2.
5. On any turn, the player whose turn it is may call Truco/Seis (per §7) **instead of** playing a card, before they play.
6. Resolve the round per §5.4 and award points per §6.
7. Swap the leader assignment for the next round and re-deal.

## 9. Information Visibility (for the RL environment)

- Each player sees: their own 3 cards, the vira, the public play history (cards played, calls made, who led each trick), the current score, the current stake, and whose turn it is.
- Each player does **not** see: the opponent's hidden cards or any cards remaining undealt.

---

---

## 10. Four-Player Variant (2 vs 2)

The 4-player variant uses the same deck, card ranking, manilha mechanics, scoring (race to 12), and Truco/Seis stake structure as the 2-player game. The differences are: team play, a card-passing phase, open team communication, and a tweaked trick-resolution rule.

### 10.1 Teams and Seating

- **4 players** form **2 teams of 2**.
- Players sit so that **teams alternate around the table**. Counterclockwise seat order: **A1, B1, A2, B2** (where A* are on Team A and B* are on Team B). Teammates sit across from each other.
- Match is still **first team to 12 points**.

### 10.2 Dealing and the Card Pass

Each round proceeds as follows:

1. The dealer rotates each round (standard counterclockwise rotation).
2. Determine who leads trick 1 — **the lead seat rotates counterclockwise by 1 each round** (so over 4 rounds, every seat leads exactly once).
3. For each team, identify the **team leader for this round**: the team member who is *earlier in the turn order* for trick 1. (Example: if lead order is B1 → A2 → B2 → A1, then Team A's leader is A2 and Team B's leader is B1.)
4. Deal cards face down:
   - **Each team leader receives 4 cards.**
   - **Each non-leader receives 2 cards.**
   - The **vira** is flipped face up to determine the manilha rank, exactly as in §4.
5. **Card pass:** each team leader looks at their 4 cards, **keeps 3**, and **passes 1 face-down to their teammate**.
   - The passed card is **known to both members of that team** (the leader saw it; the teammate looks at it on receipt).
   - The passed card is **unknown to the opposing team**.
   - The pass is mandatory — every leader must pass exactly one card.
   - Both leaders pass simultaneously (or in any order — they cannot react to each other's pass since contents are hidden).
6. After the pass, every player holds **3 cards** for the round.

### 10.3 Communication ("Chiba")

- Players on the **same team may talk freely** at any time during the round — describe their cards, lie about their cards, use signals, gestures, codes, anything.
- **Players may NOT physically show their cards** to their teammate (other than what is implicitly shared via the pass). All actual card identities remain hidden information.
- Talk is **public** — the opposing team hears everything (and may speak too).
- The card the leader passed is **shared private information** between the two teammates and serves as a reliable signal; the surrounding talk is **cheap talk** (potentially deceptive).
- This is the most distinctive feature of the variant and the core of its game theory.

### 10.4 Turn Order

- Within a trick, play proceeds **counterclockwise** starting from the trick's leader. Each player plays exactly one card per trick.
- **Trick 1's leader** is the seat determined in §10.2 step 2.
- **Subsequent tricks** are led by the player who won the previous trick. Play still proceeds counterclockwise from that leader.
- After an empate trick, the **player who led the empate trick leads again** (same as §5.3).

### 10.5 Trick Resolution (Highest Card per Team)

Once all 4 players have played a card on a trick:

1. For each team, identify the **higher of the two cards that team played** (under the §3 ranking).
2. Compare those two team-best cards.
   - The team with the **strictly higher** team-best card **wins the trick**.
   - If the two team-best cards are of **equal effective rank** → **empate** (no team wins the trick).
3. The trick is credited to the winning team (not to a specific player); however, the **specific player who played the winning card** leads the next trick.
4. If the trick is an empate, the player who **led that empate trick** leads the next one.

### 10.6 Round Outcome

A team wins the round once they win **2 of the 3 tricks**, or once all 3 tricks have been played. End-of-round resolution mirrors §5.4 applied to **team trick wins**:

- If one team has more trick wins, they win the round.
- If trick wins are tied, the round goes to the team that **won the earliest non-tied trick**.
- If all 3 tricks are empates, the round goes to the team **whose player led trick 1**.

The winning team is awarded the round's current stake (1, 3, or 6), added to their team score.

### 10.7 Truco and Seis (Team Version)

The stake schedule (1 → 3 → 6, no further raises) is identical to §7. Differences in 4-player:

- **Who can call:** any player, on their own turn, before they play a card on the current trick. Either teammate may issue the call.
- **Open team consult on response:** when a Truco or Seis is called, **the responding team may look at each other's 3 cards in private** (i.e., both teammates examine each other's hands away from the calling team) before deciding to **accept, fold, or re-raise (Seis only)**.
   - This consult window applies **only to the responding team** and **only at the moment of responding** — the calling team does not get to look at each other's hands as a result of their own call.
   - After the response is given, hands return to being held privately (each player still only sees their own cards from then on, plus whatever they remember).
   - The consult is implicit shared knowledge between teammates from that point forward (they have seen each other's hands).
- **Response is a team decision:** either responding teammate may speak the response on behalf of the team; the team may discuss openly during the consult.
- **Fold payouts and re-raise structure are unchanged** from §7. Folding gives the calling team the previous stake.

### 10.8 Information Visibility (4P RL Environment)

Each player can observe:

- Their own 3 cards.
- The vira.
- If they are a team leader: the identity of the card they passed to their teammate.
- If they are a non-leader: the identity of the card their leader passed to them (i.e., one specific card in their hand is known to be from the leader).
- All public play history: cards played on tricks, who played them, who led each trick, all calls (Truco/Seis) and responses, current score and stake.
- All public communication: anything said by any player at the table.
- After their team has responded to a Truco/Seis call: the contents of their teammate's hand (from that consult onward).

Each player does **not** directly observe:

- The opposing team leader's pass (its identity).
- The opposing team's cards (unless inferable from play or talk).
- Their own teammate's other 2 cards, unless their team has been called on (see consult rule) or unless inferable from talk and play.

> **Note for RL:** the talking channel is open and unstructured. In practice, agents will need some action space for messages (free-form, a fixed vocabulary of signals, or a learned discrete code). This is a design choice for the environment, not a rule of the game.

---

## Glossary

- **Vira** — the face-up card dealt after the hands, used to determine the manilha rank.
- **Manilha** — a card of the trump rank (the rank above the vira). Manilhas beat all non-manilha cards.
- **Trick** — one exchange in which each player plays one card; usually one player wins it.
- **Empate** — a tied trick (neither player wins).
- **Round / hand** — the sequence of up to 3 tricks played from a single deal.
- **Stake** — the point value the current round will pay to its winner (1, 3, or 6).
- **Truco** — a call that raises the stake from 1 to 3.
- **Seis** — a call that raises the stake from 3 to 6.
- **Team leader (4P)** — the team member who plays earlier in trick 1's turn order; receives 4 cards at deal and passes 1 to their teammate.
- **Card pass (4P)** — the leader's mandatory hand-off of one card to their teammate; the card's identity is shared between teammates only.
- **Chiba (4P)** — open table talk; players may say anything, true or false, during the round.
- **Consult (4P)** — the private look at each other's hands that the responding team gets when a Truco/Seis is called against them.
