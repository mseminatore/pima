---
title: "An Analysis of the Pima Indian Diabetes Dataset"
author: Mark Seminatore
date: `{clay} Date now format("%m-%d-%Y")`
---

# Introduction

This is a report summarizing a data analysis conducted on the benchmark machine learning
dataset used to predict whether a patient has diabetes based on a collection
of diagnostic measurements. Originally compiled by the National Institute of 
Diabetes and Digestive and Kidney Diseases (NIDDK), it contains data for **768** 
female patients aged 21 and older of Pima Indian heritage living near Phoenix, Arizona. 

>Note: The dataset for this analysis was sourced from 
>[Kaggle.com](https://www.kaggle.com/datasets/kumargh/pimaindiansdiabetescsv).

# Analysis Tools

This analysis was conducted using a NN model written in the [Clay](https://github.com/mseminatore/clay) programming language. Clay is a new prototype-based, pure Object-Oriented language. It runs interpretively like Python and thus allows for rapid analysis of data. Like Python, it has a growing collection of packages (called bricks) that are useful for data analysis. The Clay language shares many traits with Smalltalk and Io.

A Clay "brick" (a package) called [Clay-ann](https://github.com/mseminatore/clay-ann) provided the NN framework. Clay-ann is a native code brick built on the C neural network library [libann](https://github.com/mseminatore/ann).

This report was generated using the [Clay-knit](https://github.com/mseminatore/clay-knit) brick, which itself uses the [Clay-pdf](https://github.com/mseminatore/clay-pdf) and [Clay-plot](https://github.com/mseminatore/clay-plot) bricks. `Knitting` reports is a concept inspired by the R programming language where a markdown file contains snippets of code that are intermixed with text narrative. The code is executed in chunks with the output visible and available for generating inline tables and figures.

# The Dataset

The Pima dataset describes the medical records for a sampling of Pima Indians. For each patient it records a number of diagnostic features and whether or not they had an onset of diabetes within five years.

The data contains a total of 8 features and one class variable (0: negative and 1: positive for diabetes). The 8 features provided in the dataset include:

```clay
import("ann")

features = [
    "Preg",
    "Gluc",
    "BP",
    "Skin",
    "Insulin",
    "BMI",
    "Pedigree",
    "Age"
]

feature_count = features length

// cols where 0 means "missing": Glucose, BloodPressure, SkinThickness, 
// Insulin, and BMI. 0 is a legitimate value for Pregnancies, and 
// Pedigree/Age are never 0.

missingCols = [1, 2, 3, 4, 5]
```

The units and detailed descriptions of these features are shown in @tbl-features. Of these, only the number of pregnancies can validly be `0`. For all others, a `0` value marks missing data.

| Col | Name | Description |
|-----|------|-------------|
|1| preg | Number of times pregnant |
|2| gluc | Plasma glucose concentration 2 hours into an oral glucose tolerance test |
|3| BP | Diastolic blood pressure (mm Hg) |
|4| skin | Triceps skin fold thickness (mm) |
|5| test | 2-Hour serum insulin (mu U/ml) |
|6| BMI | Body mass index (weight in kg/(height in m)^2) |
|7| pedi | Diabetes pedigree function |
|8| age | Age (years) |
Table: Feature labels. {#tbl-features}

## Loading the data

The complete dataset is small and is stored in a single .csv file of about 23 KBytes. Feature data is in the first 8 columns. The output class is the 9th column.

```clay
    // load the Pima dataset
    "Loading Pima dataset..." println
    data = Ann loadCsv("data/pima-indians-diabetes.csv")

    feature_count = features length
    "Data loaded with ${data cols} cols and ${data rows} rows" println
```

The first thing to note from the data is that the two output classes are not evenly balanced. As @fig-balance shows, there are 500 negative and 268 positive patients. Therefore, a model that always predicted "no diabetes" would already be right about 65% of the time. As a result, the final accuracy should be judged against that baseline.

```{clay #fig-balance caption="Number of patients in each class."}
class_counts = [0, 0]

data toArray each { |row|
    // label loads as a float (1.0), so compare it rather than using it as an index
    k = ((row at(feature_count)) == 1) ifTrue { 1 } ifFalse { 0 }
    class_counts setAt(k, (class_counts at(k)) + 1)
}

Plot figure(#{figsize: [4, 3.5]})
class_bars = Plot bar(["No diabetes", "Diabetes"], class_counts, #{color: ["steelblue", "tomato"]})

Plot barLabel(class_bars, #{fmt: "%d"})
Plot ylabel("Patients")
Plot title("Classes are not balanced")
```

## Cleaning the data

As @fig-zeros below clearly demonstrates, there are a number of missing data entries. These missing entries must be dealt with before training to avoid skewing the results.

```{clay #fig-zeros caption="Number of 0 entries in each feature column."}
rows = data toArray
zeros = features map { |name| 0 }

rows each { |row|
    feature_count times { |c|
        ((row at(c)) == 0) ifTrue { zeros setAt(c, (zeros at(c)) + 1) }
    }
}

Plot figure(#{figsize: [7, 3.5]})
bars = Plot bar(features, zeros, #{color: "steelblue"})
Plot barLabel(bars, #{fmt: "%d"})
Plot ylabel("Rows with 0")
Plot title("Missing data by feature")
```

The Insulin column is missing the most data with `{clay} zeros at(features indexOf("Insulin"))` empty rows. That represents `{clay} (zeros at(4)) / (data rows)`% of all rows.

The distribution of each feature (raw, before cleaning) is shown in @fig-hist. The spikes at 0 in Glucose, BloodPressure, SkinThickness, Insulin and BMI are the missing values.

```{clay #fig-hist caption="Distribution of each feature before cleaning."}
hist_rows = data toArray
[hist_fig, hist_axs] = Plot subplots(2, 4, #{figsize: [9, 4.5]})

feature_count times { |i|
    hist_r = (i >= 4) ifTrue { 1 } ifFalse { 0 }
    hist_c = i - 4 * hist_r
    hist_ax = (hist_axs at(hist_r)) at(hist_c)
    hist_vals = hist_rows map { |row| row at(i) }
    hist_ax hist(hist_vals, 20, #{color: "steelblue"})
    hist_ax title(features at(i), #{fontsize: 9})
}
```

Splitting the same distributions by outcome, in @fig-byclass, shows which features actually separate the two groups (the missing-value zeros are left out here). Glucose, and to a lesser degree BMI and Age, shift clearly to the right for diabetic patients, while BloodPressure and SkinThickness barely differ.

```{clay #fig-byclass caption="Feature distributions by outcome (missing values excluded)."}
[cls_fig, cls_axs] = Plot subplots(2, 4, #{figsize: [9, 4.5]})

feature_count times { |i|
    cls_r = (i >= 4) ifTrue { 1 } ifFalse { 0 }
    cls_c = i - 4 * cls_r
    cls_ax = (cls_axs at(cls_r)) at(cls_c)
    cls_neg = []
    cls_pos = []
    cls_missing = missingCols contains(i)

    hist_rows each { |row|
        v = row at(i)
        ((v != 0) || (cls_missing == false)) ifTrue {
            ((row at(feature_count)) == 1) ifTrue { cls_pos push(v) } ifFalse { cls_neg push(v) }
        }
    }

    cls_ax hist(cls_neg, 15, #{color: "steelblue", alpha: 0.6, label: "No diabetes"})
    cls_ax hist(cls_pos, 15, #{color: "tomato", alpha: 0.6, label: "Diabetes"})
    cls_ax title(features at(i), #{fontsize: 9})
    (i == 0) ifTrue { cls_ax legend(#{fontsize: 7}) }
}
```

To analyze the dataset, given many missing values, we need to either drop those rows (losing precious data from such a small dataset), or better, find some way to replace the missing data.

One way to replace, or impute, missing data is to fill empty data columns with the column mean or median. This analysis selected mean.

Several helper functions were created to find the mean
value of all training_data columns and to use that to replace missing data fields.

```clay
    // mean of the non-zero values in each column. Fitted on the training data only,
    // so nothing leaks in from the validation/test sets.
    imputeMeans = { |train_data|
        rows = train_data toArray
        means = []
        train_data cols times { |c|
            sum = 0
            n = 0
            rows each { |row|
                v = row at(c)
                (v != 0) ifTrue { sum = sum + v; n = n + 1 }
            }
            means push((n > 0) ifTrue { sum / n } ifFalse { 0 })
        }

        "Column avgs: ${means}" println

        means
    }

    // replace 0s in the missing-data columns with the given column means
    impute = { |data, means|
        rows = data toArray
        rows each { |row|
            missingCols each { |c|
                ((row at(c)) == 0) ifTrue { row setAt(c, means at(c)) }
            }
        }
        AnnData fromArray(rows)
    }

    // impute and standardize every split using statistics from the 
    // training split only
    clean = { |train_data, val_data, test_data|
        "Cleaning data (impute 0s, standardize)..." println
        means = imputeMeans(train_data)
        train_data = impute(train_data, means)
        val_data = impute(val_data, means)
        test_data = impute(test_data, means)

        // z-score the feature columns, (x - mean) / std, with the training split's
        // statistics. The label column (the last) gets mean 0 and std 1: left as it is.
        mu = train_data mean
        sigma = train_data std
        mu setAt(feature_count, 0)
        sigma setAt(feature_count, 1)
        [train_data standardize(mu, sigma), val_data standardize(mu, sigma), test_data standardize(mu, sigma)]
    }
```

## Splitting the data

Now that we have a method for cleaning the data, we can split the full dataset into training, validation and testing groups. We use an 80/10/10 split for this analysis.

```clay
    // split into 80% train, 10% validation (early stopping), 10% test
    split = { |data|
        [train_data, rest] = data split(0.8)
        [val_data, test_data] = rest split(0.5)
        [train_data, val_data, test_data]
    }

    [train_data, val_data, test_data] = split(data)
    [train_data, val_data, test_data] = clean(train_data, val_data, test_data)

    x_train = train_data columns(0, feature_count)
    y_train = train_data column(feature_count)

    x_val = val_data columns(0, feature_count)
    y_val = val_data column(feature_count)

    x_test = test_data columns(0, feature_count)
    y_test = test_data column(feature_count)

```

## Feature correlations

Before modelling, it helps to see how the features relate to each other and to the outcome. @fig-corr shows the Pearson correlation of every pair of columns in the cleaned training data (the last row and column is the diabetes outcome). Glucose has by far the strongest link with the outcome, and several features move together (age and pregnancies, skin thickness and BMI).

```{clay #fig-corr caption="Correlation matrix of the features and the outcome (training split)."}
corr_names = features + ["Outcome"]
corr_rows = train_data toArray
corr_n = corr_names size
corr_count = corr_rows size

// each column's mean and standard deviation (over all corr_count rows, as Pearson's r needs)
corr_mean = train_data mean
corr_sd = train_data std

corr_matrix = []
corr_n times { |a|
    line = []

    corr_n times { |b|
        cov = 0
        corr_rows each { |row|
            cov = cov + ((row at(a)) - (corr_mean at(a))) * ((row at(b)) - (corr_mean at(b)))
        }
        line push((cov / corr_count) / ((corr_sd at(a)) * (corr_sd at(b))))
    }

    corr_matrix push(line)
}

Plot figure(#{figsize: [7, 6]})
corr_image = Plot imshow(corr_matrix, #{cmap: "coolwarm", vmin: -1, vmax: 1})
Plot colorbar(corr_image, #{label: "Pearson r"})
Plot xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], corr_names, #{rotation: 45})
Plot yticks([0, 1, 2, 3, 4, 5, 6, 7, 8], corr_names)

corr_n times { |a|

    corr_n times { |b|
        Plot text(b, a, (Math round((corr_matrix at(a) at(b)) * 100) / 100.0) toString, #{ha: "center", va: "center", fontsize: 7})
    }
}

Plot title("Feature correlations")
```

# Model

Given the 8 input features and 1 output class, and a presumption that the problem was linear-logistic, a simple dense network model was chosen. Since this is a single class output predictor, the loss function was set to `cross_entropy`, which `libann` treats as `Binary Cross Entropy`.

For reference, the loss function for Binary Cross Entropy is:

$$BCE = -\frac{1}{N} \sum_{i=1}^{N} [ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) ]$$

## Model details

The dataset is not large, or particularly complex. Thus, a deep NN is not necessary, and only a few small hidden layers will be used. Note that it is likely that an even smaller model may converge as well or better due to the possibility of over-fitting such a small dataset.

> In general the number of parameters in a model should be smaller than the dataset
> size to avoid over-training and to force generalized learning.

The total parameter count is modest and the model trains very quickly. The following properties were used to configure the network model:

- Validation data with `EarlyStopping` with a `patience` of 10 epochs
- The `Adam` optimizer with the default learning rate of 0.001
- Two (2) dense hidden layers with 16 and 8 nodes and **ReLU** activation
- A single dense output layer with **Sigmoid** activation

```clay
    net = Ann network(#{"earlyStopping": 10, "loss": "cross_entropy", "quiet": true})
    net input(feature_count)
    net dense(feature_count * 2, "relu")
    net dense(feature_count, "relu")
    net output(1, "sigmoid")

    net summary println
```

## Training

Having cleaned the data and defined our network model it is time to train against the data. We use validation data to determine how well the model performs against data it has not seen. This also enables early stopping when validation loss fails to improve.

```clay
    // fraction of rows classified correctly, thresholding the sigmoid output at 0.5.
    // (net accuracy() argmaxes over output columns, which is always "class 0" with a single output.)
    accuracy = { |x, y|
        xs = x toArray
        ys = y toArray
        correct = 0
        xs size times { |i|
            p = net predict(xs at(i))
            guess = ((p at(0)) >= 0.5) ifTrue { 1 } ifFalse { 0 }
            (guess == ((ys at(i)) at(0))) ifTrue { correct = correct + 1 }
        }
        (correct * 1.0) / xs size
    }

    train = { |x_train, y_train, x_val, y_val|
        net validation(x_val, y_val)
        net train(x_train, y_train)
    }

    // train on the data
    train(x_train, y_train, x_val, y_val)

    // report results
    "Training completed" println
    "------------------" println
    "Training accuracy:   ${accuracy(x_train, y_train) * 100}%" println
    "Validation accuracy: ${accuracy(x_val, y_val) * 100}%" println
    "Test accuracy:       ${accuracy(x_test, y_test) * 100}%" println

```

## Learning curve

As @fig-learn below shows, the training loss continues to show improvement. However, the validation loss levels out at around epoch 54. This is a clear indication of the start of over-training, and that the network is beginning to memorize the training data.

Displaying the learning curve to evaluate training progress requires just a few lines of Clay code.

```{clay #fig-learn caption="Learning curve."}
    history = net history

    Plot learningCurve(history)
    Plot title("Learning curve")
```

# Results

The test split (10% of the total rows) was held out of both training and early stopping, so the figures below are an honest estimate of how the model performs on unseen patients. The network's output is a probability, thresholded at 0.5 to make a prediction.

## Confusion matrix

@fig-confusion counts the test predictions against the true outcomes. For a screening task like this, the false negatives (diabetic patients the model misses, bottom left) matter more than the raw accuracy.

```{clay #fig-confusion caption="Confusion matrix on the test split (rows: actual, columns: predicted)."}
test_probs = (net predictBatch(x_test)) toArray
test_truth = y_test toArray

tn = 0
fp = 0
fn = 0
tp = 0

test_probs size times { |k|
    guess = (((test_probs at(k)) at(0)) >= 0.5) ifTrue { 1 } ifFalse { 0 }
    actual = (test_truth at(k)) at(0)
    ((actual == 0) && (guess == 0)) ifTrue { tn = tn + 1 }
    ((actual == 0) && (guess == 1)) ifTrue { fp = fp + 1 }
    ((actual == 1) && (guess == 0)) ifTrue { fn = fn + 1 }
    ((actual == 1) && (guess == 1)) ifTrue { tp = tp + 1 }
}

Plot figure(#{figsize: [4.5, 4]})
Plot confusionMatrix([[tn, fp], [fn, tp]], #{labels: ["No diabetes", "Diabetes"], title: "Test set"})
```

## ROC curve

The confusion matrix depends on the 0.5 threshold. The ROC curve in @fig-roc shows the trade-off between true and false positives at every threshold, and the area under it (AUC) summarizes the model's ability to rank positive patients above negative ones: 0.5 is chance, 1.0 is perfect.

```{clay #fig-roc caption="ROC curve on the test split."}
test_scores = test_probs map { |p| p at(0) }
test_labels = test_truth map { |r| r at(0) }

Plot figure(#{figsize: [5, 4.5]})
roc_result = Plot roc(test_scores, test_labels, #{chance: true, title: "ROC curve"})
```

## Predicted probabilities

@fig-probs splits the test predictions by true outcome. A good model pushes the two groups to opposite ends; the overlap in the middle is where the errors come from, and it is why moving the threshold away from 0.5 trades false negatives for false positives.

```{clay #fig-probs caption="Predicted probability of diabetes on the test split, by true outcome."}
probs_neg = []
probs_pos = []
test_scores size times { |k|
    ((test_labels at(k)) == 1) ifTrue { probs_pos push(test_scores at(k)) } ifFalse { probs_neg push(test_scores at(k)) }
}
Plot figure(#{figsize: [6, 3.5]})
Plot hist(probs_neg, 10, #{range: [0, 1], color: "steelblue", alpha: 0.6, label: "No diabetes"})
Plot hist(probs_pos, 10, #{range: [0, 1], color: "tomato", alpha: 0.6, label: "Diabetes"})
Plot axvline(0.5, #{color: "gray", linestyle: "--"})
Plot xlabel("Predicted probability")
Plot ylabel("Patients")
Plot legend
Plot title("Predicted probabilities")
```

## Feature importance

Which inputs does the network actually rely on? @fig-importance measures it by permutation: each feature's column in the test split is shifted against the outcomes, breaking its link with them, and the drop in accuracy is recorded (averaged over four different shifts). A bigger drop means the model leans on that feature more.

A bar near zero or slightly negative means the model does not use the feature. With only 77 test patients, small differences like these are noise (a shuffle can happen to help by chance), so read the large bars, not the small ones: Glucose and Pregnancies stand out.

```{clay #fig-importance caption="Permutation importance: accuracy lost when a feature is shuffled (test split)."}
imp_rows = x_test toArray
imp_n = imp_rows size
imp_base = accuracy(x_test, y_test)
imp_shifts = [7, 19, 31, 43]

imp_drops = []
feature_count times { |f|
    total = 0
    imp_shifts each { |shift|
        shifted = []
        imp_n times { |r|
            row = []
            feature_count times { |c|
                src = (c == f) ifTrue { (r + shift) % imp_n } ifFalse { r }
                row push((imp_rows at(src)) at(c))
            }
            shifted push(row)
        }
        total = total + (imp_base - accuracy(AnnData fromArray(shifted), y_test))
    }
    imp_drops push(total / imp_shifts size)
}

Plot figure(#{figsize: [6, 4]})
Plot barh(features, imp_drops, #{color: "steelblue"})
Plot xlabel("Drop in accuracy")
Plot title("Permutation importance")
```

## What the network learned

Finally, @fig-embed projects the test patients to two dimensions with PCA, colored by outcome. On the left are the 8 standardized input features. On the right is the output of the last hidden layer, which is what the network hands to its output neuron. The hidden layer arranges the patients along the direction the network uses to decide: diabetic patients gather toward the left, negative patients toward the right. 

The overlap that remains in the middle in both panels reflects the limit of what these 8 measurements can say about diabetes.

```{clay #fig-embed caption="PCA of the test patients: input features (left) and last hidden layer (right)."}
embed_classes = y_test toArray map { |r| ((r at(0)) == 1) ifTrue { 1 } ifFalse { 0 } }
embed_hidden = (net activations(x_test, -2)) toArray

[embed_fig, embed_axs] = Plot subplots(1, 2, #{figsize: [9, 4]})
embed_axs at(0) embedding(x_test toArray, embed_classes, #{classNames: ["No diabetes", "Diabetes"], s: 14, alpha: 0.7})
embed_axs at(0) title("Input features")
embed_axs at(1) embedding(embed_hidden, embed_classes, #{classNames: ["No diabetes", "Diabetes"], s: 14, alpha: 0.7})
embed_axs at(1) title("Last hidden layer")
```

# Quantization

A model this small is a natural fit for a microcontroller or a phone, where memory and floating-point hardware are scarce. Clay-ann can shrink a trained network in two ways:

- `quantize` makes an **int8** copy of the whole network (weights and activations), so it can run with integer arithmetic alone.
- `weightFormat` stores the hidden layers' weights in **int4**, **ternary** (2 bits: -1, 0 or +1 times a scale) or **binary** (1 bit: ± a scale). The output layer stays float.

Rounding weights this coarsely loses information, so each low-bit format is tested twice: rounded straight after training, and again after 10 more epochs of *quantization-aware* fine-tuning, in which the network trains through the rounded weights and adjusts to them. Every format starts from a fresh copy of the float network trained above, and is scored on the same 77 held-out test patients by accuracy (threshold 0.5) and by ROC AUC.

```clay
    // score a model on the test split: [accuracy, AUC]
    score = { |m|
        q_scores = (x_test toArray) map { |row| (m predict(row)) at(0) }
        q_correct = 0
        q_scores size times { |k|
            q_guess = ((q_scores at(k)) >= 0.5) ifTrue { 1 } ifFalse { 0 }
            (q_guess == (test_labels at(k))) ifTrue { q_correct = q_correct + 1 }
        }
        [(q_correct * 1.0) / q_scores size, (PlotML rocCurve(q_scores, test_labels)) get("auc")]
    }

    // the trained float network, saved so each format starts from a copy of it
    q_path = File tempDir + "/pima-float.net"
    net save(q_path)
    q_calib = x_train slice(0, 500)          // rows that set the int8 value ranges

    q_formats = ["float32", "int8", "int4", "ternary", "binary"]
    q_bits = [32, 8, 4, 2, 1]
    q_rounded = [score(net), score(net quantize(q_calib))]
    q_tuned = [nil, nil]

    ["int4", "ternary", "binary"] each { |fmt|
        m = Ann load(q_path)
        m weightFormat(fmt)
        q_rounded push(score(m))
        m fit(x_train, y_train, #{"epochs": 10})
        q_tuned push(score(m))
    }
    File delete(q_path)

    // weights in the two hidden layers (8x16 and 16x8)
    q_weights = feature_count * feature_count * 4

    q_formats size times { |i|
        line = "${q_formats at(i)}: rounded ${q_rounded at(i)}"
        ((q_tuned at(i)) != nil) ifTrue { line = line + ", fine-tuned ${q_tuned at(i)}" }
        line println
    }
```

@fig-quant compares the formats. The hidden layers hold `{clay} q_weights` weights, so they take `{clay} q_weights * 4` bytes as float32, `{clay} q_weights` as int8, `{clay} q_weights / 2` as int4, `{clay} q_weights / 4` as ternary and only `{clay} q_weights / 8` as binary.

```{clay #fig-quant caption="Test accuracy and AUC by weight format, rounded after training and after quantization-aware fine-tuning. The dashed line on the left is the 65% majority-class baseline."}
[q_fig, q_axs] = Plot subplots(1, 2, #{figsize: [9, 3.8]})
q_pos = [0, 1, 2, 3, 4]
q_names = ["Accuracy", "AUC"]

2 times { |j|
    q_ax = q_axs at(j)
    q_ax bar([0, 1, 1.8, 2.8, 3.8], q_rounded map { |s| s at(j) }, #{width: 0.4, color: "steelblue", label: "Rounded"})
    q_ax bar([2.2, 3.2, 4.2], [2, 3, 4] map { |i| (q_tuned at(i)) at(j) }, #{width: 0.4, color: "tomato", label: "Fine-tuned"})
    q_ax xticks(q_pos, q_formats)
    q_ax ylim(0.5, 1)
    q_ax title(q_names at(j))
    (j == 0) ifTrue {
        q_ax axhline(0.65, #{color: "gray", linestyle: "--"})
        q_ax legend(#{fontsize: 8})
    }
}
```

The int8 network scores the same as the float one: 8 bits are plenty for these weights, so int8 cuts the weights to a quarter of their size for free. int4 also holds up, losing a little AUC when rounded and getting it back with fine-tuning. Ternary and binary weights are a different story: rounded straight after training, they fall to or near the 65% baseline, little better than always answering "no diabetes". Fine-tuning recovers most of the loss, and a fine-tuned binary network keeps an AUC above 0.8 with its hidden weights in just 32 bytes.

With only 77 test patients, one patient is 1.3 points of accuracy, so differences of a point or two between formats (such as int4 edging out float32) are noise. The large drops before fine-tuning are not.
