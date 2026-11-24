# Data dictionary — `msme_digital_readiness_anonymized.csv` (not published)

Cross-sectional survey of 159 traders in three traditional markets in Ternate, Indonesia (Gamalama, Higienis, Bastiong), collected between May and July 2026 with assisted Google Forms and paper questionnaires after verbal informed consent. The file is available from the corresponding author on reasonable request (amalkhairan@unkhair.ac.id). Place it in this folder to run the code. Answers are stored as the original Indonesian option texts; `code/encode.py` converts them to scores.

Row order is the original order of the survey export and must be kept: the cross-validation splits (`random_state=42`) depend on it.

## Columns

| Column | Content |
| --- | --- |
| `respondent_id` | Sequential identifier 1–159 (no link to any personal data) |
| `Jenis Kelamin` | Gender |
| `Pendidikan Terakhir` | Highest education |
| `Usia` | Age band |
| `Jenis Usaha/dagangan` | Business type |
| `Lokasi Pasar` | Market |
| `Lama Jualan` | Years in business (band) |
| `RataRata Omzet per BUlan` | Average monthly turnover (band) |
| `1.` … `24.` | Questionnaire items Q1–Q24, five ordered options each (scored 1–5 by `encode.py`) |

## Label

Digital Readiness Score = Q8 + Q9 + Q10 + Q12 + Q14 (range 5–25). Classes: Low 5–11 (n = 108), Medium 12–18 (n = 42), High 19–25 (n = 9). The five label items are excluded from the predictors.

## Removed from the anonymized file

Response timestamp, e-mail address, respondent name and telephone number.
