# pima

A small neural network that predicts the onset of diabetes from the
[Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/kumargh/pimaindiansdiabetescsv),
written in [Clay](https://github.com/mseminatore/clay) with the
[clay-ann](https://github.com/mseminatore/clay-ann) brick. The analysis is a knitted report:
[`report.md`](report.md) which mixes the narrative text with Clay code that loads, cleans, and trains a NN model using the data. Then, [clay-knit](https://github.com/mseminatore/clay-knit) turns it into a PDF or web page with the output and figures.

## The data

768 female patients, aged 21 or older, of Pima Indian heritage, from the National Institute of
Diabetes and Digestive and Kidney Diseases. Each row has 8 diagnostic measurements (pregnancies,
glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree, age) and whether the
patient developed diabetes within five years. The classes are unbalanced (500 negative, 268
positive), so always guessing "no diabetes" is already right 65% of the time.

## The model

- Zeros in glucose, blood pressure, skin thickness, insulin and BMI mark missing values. They are
  replaced with the training split's column means, then every feature is standardized. Both steps
  use statistics from the training split only.
- An 80/10/10 split into training, validation (for early stopping) and test rows.
- A dense network, 8 → 16 → 8 → 1 (ReLU hidden layers, sigmoid output), trained with Adam and
  binary cross-entropy, with early stopping after 10 epochs without improvement.

## Results

On the 77 held-out test patients (with `Ann seed(1)`), the network is right 77.9% of the time,
against the 65% baseline, with a ROC AUC of 0.886. The report adds a confusion matrix, ROC curve,
permutation feature importance (glucose matters most), and a PCA view of what the last hidden
layer learned.

### Quantized models

The report also tests how well the network survives being shrunk for small devices. Each format
is scored straight after rounding, and again after 10 epochs of quantization-aware fine-tuning:

| Format | Bits per weight | Accuracy (rounded → fine-tuned) | AUC (rounded → fine-tuned) |
|---|---|---|---|
| float32 | 32 | 77.9% | 0.886 |
| int8 | 8 | 77.9% | 0.885 |
| int4 | 4 | 77.9% → 79.2% | 0.872 → 0.889 |
| ternary | 2 | 66.2% → 76.6% | 0.788 → 0.863 |
| binary | 1 | 68.8% → 74.0% | 0.699 → 0.827 |

int8 and int4 cost nothing. Ternary and binary weights drop to near the 65% baseline until the
network is fine-tuned, which recovers most of the loss. With 77 test patients, one patient is 1.3
points of accuracy, so read only the large differences.

## Running it

You need [Clay](https://github.com/mseminatore/clay). From this folder:

```sh
clay bricks install          # fetches clay-ann, clay-knit and their dependencies into bricks/
clay src/main.clay           # trains the model and prints its accuracy
clay knit.clay               # knits report.md into report.pdf
clay knit.clay --html        # or into report.html, with interactive figures
```

## Layout

| Path | |
|---|---|
| `report.md` | the analysis: narrative and Clay code chunks |
| `knit.clay` | knits the report |
| `src/main.clay` | the same pipeline as a standalone program |
| `data/pima-indians-diabetes.csv` | the dataset |
