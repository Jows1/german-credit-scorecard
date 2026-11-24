# Credit Application Scorecard & Cut-off Strategy (German Credit)

A credit scorecard that rates loan applicants from 1,000 historical applications, and turns the scores into a practical **approve / refer / decline** lending policy. Built the way a credit risk team would: Weight of Evidence (WoE), Information Value (IV), logistic regression, points scaling, and a profit-based cut-off analysis.

**Tools:** Python (pandas, NumPy, scikit-learn) in a Jupyter notebook · **Data:** [German Credit dataset](https://www.openml.org/d/31) (UCI Statlog, loaded via OpenML)

---

## Key results

| | Result |
|---|---|
| Model performance (unseen test set) | **Gini 0.57 · AUC 0.78 · KS 0.49** |
| Score separation | Average score: goods **531**, bads **494** (≈3.6× better odds for the typical good applicant) |
| Most profitable single cut-off | **530**: approves 40.7% of applicants with a **7.6%** bad rate |
| Recommended policy | **Approve ≥ 530 · Refer 500–529 for manual review · Decline < 500** |
| Without a scorecard (approve everyone) | 30% bad rate and a loss of **−800 profit units**, vs **+221** at the best cut-off |

---

## 1. Data check

- 1,000 applicants, 20 features, target `class` (good / bad). **Bad rate: 30%.** No missing values.
- Split into **train (800)** and **test (200)** *before* any analysis, stratified so both keep a 30% bad rate. All bands and WoE values are learned on train only.

**First look (training data):**

| Checking account | Applicants | Bad rate |
|---|---|---|
| Overdrawn (<0) | 219 | 47.9% |
| 0–200 | 214 | 39.3% |
| ≥200 | 51 | 25.5% |
| No checking account | 316 | 12.0% |

**A result I flagged rather than trusted:** applicants with a *critical* credit history had the **lowest** bad rate (17.5%). That goes against business sense, and is most likely **selection bias**: the data only contains approved loans, so applicants with a critical history were probably approved only when everything else was strong. In a real lending policy this would go to review before letting the model reward a critical history.

## 2. Feature selection (Information Value)

| Feature | IV | Strength |
|---|---|---|
| checking_status | 0.617 | Very strong (checked: known at application time, not a leak) |
| credit_history | 0.265 | Medium |
| savings_status | 0.225 | Medium |
| duration (banded) | 0.217 | Medium |
| purpose | 0.153 | Medium |
| property_magnitude | 0.147 | Medium |
| employment | 0.125 | Medium |
| other_payment_plans | 0.090 | Weak |
| housing | 0.088 | Weak |
| credit_amount (banded) | 0.067 | Weak |
| age (banded) and 7 others | < 0.05 | Dropped |

**Rule:** kept features with IV ≥ 0.05, giving **10 features**: enough to predict well, few enough to explain to a credit committee.

## 3. Model

- Numeric features (duration, credit amount) split into 5 quantile bands; small `purpose` categories (< 30 applicants) merged into "other" for stable WoE.
- Each feature replaced by its WoE value, then a **logistic regression** fitted on the training set.
- **All 10 coefficients are negative**, so every feature moves risk in the expected direction.

| Metric (test set) | Value | Benchmark |
|---|---|---|
| AUC | 0.784 | — |
| Gini | 0.568 | 0.4–0.6 is solid for application scorecards |
| KS | 0.486 | 0.3–0.5 is good separation |

## 4. Scorecard

Scaled so that **600 points = 50:1 good:bad odds**, and **every 20 points doubles the odds** (PDO = 20). Each application answer earns points; the total is the score. Higher = safer.

**Extract:**

| Feature | Category | Points |
|---|---|---|
| Checking account | No checking account | 76 |
| | ≥200 | 56 |
| | 0–200 | 42 |
| | Overdrawn (<0) | 34 |
| Savings | ≥1,000 | 76 |
| | 500–1,000 | 69 |
| | <100 | 45 |
| Loan duration | ≤12 months | 64 |
| | 12–15 months | 68 |
| | >30 months | 33 |
| Credit history | Critical / other existing credit | 65 |
| | No credits / all paid | 26 |

Test-set scores range from 431 to 607 (median 517).

## 5. Cut-off strategy

**Profit assumption:** approving a bad borrower costs **5×** as much as rejecting a good one (the cost ratio given with the original dataset). Profit units = goods approved × 1 − bads approved × 5.

| Cut-off | Approval rate | Bad rate (approved) | Bads declined | Profit units |
|---|---|---|---|---|
| 480 | 84.6% | 23.0% | 35.0% | −324 |
| 490 | 77.7% | 20.2% | 47.7% | −165 |
| 500 | 68.4% | 16.5% | 62.3% | +6 |
| 510 | 60.2% | 14.1% | 71.7% | +92 |
| 520 | 49.7% | 10.5% | 82.7% | +185 |
| **530** | **40.7%** | **7.6%** | **89.7%** | **+221** |
| 540 | 33.3% | 6.3% | 93.0% | +207 |
| 550 | 25.5% | 5.1% | 95.7% | +177 |

The most profitable single cut-off (530) approves only 4 in 10 applicants, too strict for a lender that also wants growth. So the recommendation uses **three bands**:

| Band | Score | Share of applicants | Bad rate | Action |
|---|---|---|---|---|
| ✅ Approve | ≥ 530 | 40.7% | 7.6% | Approve automatically |
| 🟡 Refer | 500–529 | 27.7% | 29.6% | Manual review and field verification |
| ❌ Decline | < 500 | 31.6% | 59.2% | Decline |

The refer band's bad rate matches the portfolio average: the score can't separate these applicants well, which is exactly where an analyst's review of documents, cash flow and field verification adds the most value. This keeps the book profitable while still giving about a quarter of applicants a human assessment, **balancing growth against portfolio quality**.

---

## Decisions log

| Decision | Reason |
|---|---|
| Used the full 20-feature dataset (OpenML `credit-g`) | The popular 10-column Kaggle extract has no good/bad target |
| Dropped `personal_status` and `foreign_worker` | They encode sex, marital status and nationality, which should not drive lending decisions |
| Split train/test before any binning or WoE | Prevents information from the test set leaking into the model |
| Kept features with IV ≥ 0.05 | Balance between predictive power and explainability |
| Merged `purpose` categories with < 30 applicants | Small groups give unstable WoE |
| Unseen categories get WoE 0 (neutral points) | Safe default for new values in production |
| Three-band policy instead of one cut-off | The most profitable cut-off rejects 59% of applicants |

## Limitations and next steps

- **Small test set:** 200 applicants (60 bads), so Gini and KS would move somewhat with a different split. Cross-validation would give a more stable estimate.
- **The cut-off table is scored on all 1,000 applicants**, including the 800 used for training, so it is slightly optimistic. Bands should be confirmed on new applications.
- **Profit depends on the 5:1 cost assumption.** A lender with different margins or loss rates would choose different bands.
- **Reject inference:** the data only includes approved loans, which likely explains the counter-intuitive credit-history result.
- **Monotonicity:** the 12–15 month duration band scores slightly higher than ≤12 months. Merging those bands would make points rise smoothly with shorter loans, as business logic expects.
- **Next:** probability-of-default calibration, monitoring (population stability index), and an expected credit loss (IFRS 9) extension.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Open `scorecard.ipynb` and run all cells. The data downloads automatically from OpenML on first run and is saved to `data/`.

---

**Author:** Omoyajowo David · [GitHub](https://github.com/Jows1?tab=repositories) · [LinkedIn](https://linkedin.com/in/omoyajowo-david-56524a217)
