# 05 · Data Analysis: The Datasets, Their Defects and the Cleaning

> Files: `data/`, `data_pipeline/*.py`, `run_data_pipeline.py`, `output/data_quality_report.json`

## 1. What arrives

### Historical extracts (for training the models): dirty

| File | Rows received → kept | One row is… | The answer column |
|---|---|---|---|
| `campaign_submissions_train.csv` | 30,706 → **29,697** | one campaign's copy submission | `amendment_required` (Yes/No) |
| `campaign_submissions_eval.csv` | 8,514 → **8,224** | same, held out for scoring | same |
| `programme_audiences_train.csv` | 30,877 → **29,817** | one airing of one programme | `under18_viewers` (thousands) |
| `programme_audiences_eval.csv` | 7,004 → **6,770** | same, from ten other weeks | same |
| `clearance_notes.csv` | 34,727 → **33,045** | one free-text note about a copy | `issue_category` (8 issues) |

### The live queue (`data/input/`): what the system works on this week

| File | Size | Notes |
|---|---|---|
| `open_campaigns.csv` | 240 campaigns × 40 columns | Clean; the 37 submission columns (no answer) + `advertiser_name`, `agency_name`, `copy_title`. LHF 79, Alcohol 50, Betting 42, Medicines 39, Credit 30 |
| `booked_spots.csv` | 4,833 spots | **Exactly as each scheduler wrote it**: TXLine writes `SAT 25:10`, Airtime Planner `Saturday`, SlotMaster `Sat`, Planit `Sa.`; lengths as `30"`, `0:30`, `30s`, `00:00:30` |
| `programme_schedule.csv` | 681 programmes | Clean, already on the broadcast clock, 06:00–30:00 with no gaps; no audience |
| `open_campaign_notes.csv` | 789 notes | Every live campaign has at least one note |

## 2. The key columns, in plain words

**Submissions:**
- `clock_number` `AAA/BBBB123/030`: agency / advertiser+copy / **length in seconds**.
- `copy_length_seconds`, `supers_count` (on-screen text), `legal_text_seconds` (small print on screen).
- Claim flags: price, health, environmental, comparative.
- Offer flags: `offer_terms_shown_flag` (empty when there's no offer, a *fact* rather than a defect), `bonus_or_free_bet_flag`, `representative_apr_shown_flag` (empty outside consumer credit).
- Booking shape: spots, channels, share before 21:00, share in live sport.
- History: the advertiser's campaigns and amendments in the past year, and the agency's amendment rate.
- `desk_code`, `weekday_submitted`, `batch_code` carry **no signal** and are not model inputs.

**Audiences:** channel, scheduling system, transmission day, start and end time, genre, flags (live, live sport, made for children, premiere, school holiday), episode number (empty for films, a fact), day type, lead-in genre, `forecast_under18_index` (100 = average), `audience_all` (measured **after** airing, so never an input), `under18_viewers` (the answer).

**Notes:** author, role, source (Executive first read / Legal review / Agency reply / Executive update / Controller comment), sequence, timestamp, text, length, issue.

## 3. Interesting patterns in the data (useful in the review)

- About **29%** of submissions needed amendments, almost identical in every category (28.9%–29.8%). So the category alone doesn't predict amendment; the copy's features do.
- Copy **with a health claim** needed changes **41%** of the time vs **26%** without.
- An offer with its **terms not shown** needed changes **69%** of the time vs **35%** with terms shown.
- Programmes **made for children** have a median of **40.4k** under-18 viewers vs **9.3k** for others. Harbourlight One has the largest young audience (median 33.3k), Drama the smallest (6.8k).
- The under-18 audience is **heavily right-skewed** (median 12.2k, mean 47.2k, max 1,563.7k). That's why the exposure model uses **Poisson** loss.
- Clearance notes are **imbalanced**: Health claim 22.6%, Price/offer 21.3%, Under-18 appeal 19.1% … Environmental claim only 4.2%. That's why the issue model uses **balanced class weights** and **macro F1**.

## 4. The defects, and exactly how each is repaired

Every repair uses constants from `config/config.py`. **Anything in an unknown notation is refused with a `ValueError` naming the column and the values. Nothing is guessed.**

| Defect | Example | Repair (function) |
|---|---|---|
| Padding and case | `" Fast food "`, `"CONSUMER CREDIT"` | strip on load; mappings are lower-cased (`normalize_category_values`) |
| Spelling variants and codes | `B&G`, `HFSS`, `HL1`, `H-One`, `U18`, `Doc`, `Sitcom` | `*_MAPPING` dictionaries fold each onto one canonical value |
| Weekdays written several ways | `Sat`, `SAT`, `Sa.`, `Mo.`, `Thurs` | `WEEKDAY_MAPPING` |
| Yes/no written many ways | `Y`, `TRUE`, `1`, `N`, `FALSE`, `0` | `BOOLEAN_MAPPING` → `Yes` / `No` |
| Missing written many ways | `-`, `?`, `N/A`, `NULL`, `Not recorded`, blank | `NULL_TOKENS` → NaN on load (`load_raw_dataset`) |
| Duplicated rows | the same record twice | `remove_duplicate_records` keeps the first |
| **Two clocks** | TXLine `Saturday 25:10` vs others `Sunday 01:10` | `convert_to_broadcast_clock` (see below) |
| Small hours on the next calendar day | calendar `Sunday 05:40` | previous weekday's broadcast day, 24h later: **Saturday 29:40** |
| End written before its start | calendar `23:30–00:30` | an end at or before 06:00 is 24h later: `23:30–24:30` |
| Spot lengths in many notations | `30"`, `30s`, `30 sec`, `0:30`, `00:00:30` | `convert_spot_lengths_to_seconds` → 30 |
| Copy length disagrees with clock number | 90 s on a clock number ending `060` | **the clock number wins** (it's what playout plays) |
| Audiences in many notations | `0.2272m`, `62.3k`, `54.8 thousand`, `220,500` | `convert_audiences_to_thousands` → 227.2, 62.3, 54.8, 220.5 |
| Numbers with a unit or sign | `24 spots`, `2 channels`, `45%`, `31.0 %` | `convert_whole_numbers` / `convert_decimal_numbers` with their patterns |
| Impossible readings | index 0 or > 999, negative audience, % > 100, > 4 channels, ending after 30:00 | `treat_out_of_range_as_missing` → blank (the row is kept) |
| Contradictions | legal text longer than the copy; more amendments than campaigns | blanked (`blank_contradictions`) |
| Missing answer | no `amendment_required` | the row is removed (never given an invented answer) |
| Impossible answer | `under18_viewers = -3.2` | blanked, so the row is removed |
| Note times in many notations | `MONDAY 08:36`, `Mo. 09:42` | `Monday 08:36` via `NOTE_TIMESTAMP_PATTERN` |
| Markup / escapes in notes | `<span>`, `<p>`, `<br/>`, `&nbsp;`, `&#39;` | tags → space, `html.unescape`, single spaces (`convert_markup_to_plain_text`) |
| Notes in capitals | `FIRST READ OF 'NEW WHEELS'…` | restored in lower case |
| Notes too short | `Noted, thanks.` | removed (fewer than 6 words) |
| Stale derived column | `note_length_chars` measured on the raw text | recomputed on the cleaned text |

### The broadcast clock, the heart of the cleaning

The broadcast day runs **06:00 → 05:59 next morning**. On the broadcast clock, after midnight the hours continue as **24:00–29:59**, and a programme ends at 30:00 at the latest.

| System | Writes | Rule |
|---|---|---|
| **TXLine** (`BROADCAST_CLOCK_SYSTEMS`) | broadcast clock on the broadcast day | keep as is; a start before 06:00 is impossible → refused |
| **Airtime Planner, SlotMaster, Planit** (`CALENDAR_CLOCK_SYSTEMS`) | ordinary clock on the calendar day | a start before 06:00 moves to the **previous weekday**, +24h; an end at or before 06:00 is +24h; any time ≥ 24:00 is impossible → refused |

Examples (all checked by the tests):
- Airtime Planner `Monday 01:10` → **Sunday 25:10** (the booking week's last broadcast day).
- SlotMaster `Sa. 02:15–03:45` → **Friday 26:15–27:45**.
- Planit `Sunday 05:00–06:00` → **Saturday 29:00–30:00**.

## 5. The order of each cleaning run

**Submissions:** load as text → remove duplicates → remove rows with no answer → fold categories → read whole and decimal numbers → read copy length (refuse unknown notations) → **replace with the clock number's length** (refuse non-standard lengths) → blank out-of-range values → blank contradictions → save with the raw columns in the raw order.

**Audiences:** load → dedupe → remove no-answer rows → fold categories → **broadcast clock** → audiences to thousands → whole numbers → blank impossible readings → remove rows whose answer became blank → write times as `HH:MM` → save.

**Notes:** load → dedupe → remove no-answer rows → fold categories → read sequence → normalise timestamps → plain text → lower-case shouted notes → drop notes under 6 words → recompute length → save.

**Why only remove rows with no answer?** A gap in an ordinary column is filled *inside the model pipeline* (median or "Not recorded"), learned from training data only. Throwing such rows away would waste good data and bias the sample.

## 6. The quality report: the evidence

`output/data_quality_report.json` has one entry per extract, with general counts plus defect-specific counts. **Every removed row is accounted for:**

| Extract | Removed | = duplicates | + no answer | + other |
|---|---|---|---|---|
| submissions train | 1,009 | 706 | 303 | — |
| submissions eval | 290 | 214 | 76 | — |
| audiences train | 1,060 | 691 | 277 | 92 impossible answers |
| audiences eval | 234 | 164 | 53 | 17 impossible answers |
| notes | 1,682 | 729 | 282 | 671 too short |

Defects found as received (and **0 after cleaning**) include, for submissions train: 15,268 copy lengths not written as plain seconds, 612 lengths disagreeing with the clock number, 18 legal texts longer than the copy, and 12 cases of amendments exceeding campaigns. For notes: 6,239 with markup, 3,673 with escaped characters, 659 in capitals, and 8,472 stale lengths. For audiences train: 25,902 times on the calendar clock, 5,728 small-hours starts moved to the previous day, and 947 ends written before their starts.
