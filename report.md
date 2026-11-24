---
title: "Analysis of Pima Indian Diabetes Dataset"
author: Mark Seminatore
date: `{clay} Date now format("%m-%d-%Y")`
---

# Introduction

The is a report summarizing an analysis of the benchmark machine learning
dataset used to predict whether a patient has diabetes based on a collection
of diagnostic measuremewnts. Originally compiled by the National Institute of 
Diabetes and Digestive and Kidney Diseases (NIDDK), it contains data for **768** 
female patients aged 21 and older of Pima Indian heritage living near Phoenix, Arizona. 

>Note: The dataset for this analysis was sourced from 
>[Kaggle.com](https://www.kaggle.com/datasets/kumargh/pimaindiansdiabetescsv).

# Analysis Tools

This analysis was conducted using a NN model written in the [Clay](https://github.com/mseminatore/clay) programming language. A prototype-based pure OO language sharing traits from Smalltalk and Io.

A Clay "brick" (a package) called [Clay-ann](https://github.com/mseminatore/clay-ann) provided the NN framework. Clay-ann is a native code brick built on the C neural network library [libann](https://github.com/mseminatore/ann).

The report itself was generated using the [Clay-knit](https://github.com/mseminatore/clay-knit) brick, which itself uses the [Clay-pdf](https://github.com/mseminatore/clay-pdf) and [Clay-plot](https://github.com/mseminatore/clay-plot) bricks.

# The Dataset

The Pima dataset describes the medical records for Pima Indians and whether or not each patient will have an onset of diabetes within five years.

The data contains a total 8 features and one class variable (0: netagive and 1: positive for diabetes). The 8 features include:

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

    // columns where 0 means "missing": Glucose, BloodPressure, SkinThickness, Insulin, BMI.
    // 0 is a legitimate value for Pregnancies, and Pedigree/Age are never 0.

    missingCols = [1, 2, 3, 4, 5]
```

The units and detailed descriptions of these features are shown in @tbl-features. Of these, only number of pregnancies can validly be `0`. For all others, a `0` value marks missing data.

| Col | Name | Description |
|-----|------|-------------|
|1| preg | Number of times pregnant |
|2| plas | Plasma glucose concentration a 2 hours in an oral glucose tolerance test |
|3| pres | Diastolic blood pressure (mm Hg) |
|4| skin | Triceps skin fold thickness (mm) |
|5| test | 2-Hour serum insulin (mu U/ml) |
|6| mass | Body mass index (weight in kg/(height in m)^2) |
|7| pedi | Diabetes pedigree function |
|8| age | Age (years) |
Table: Feature labels. {#tbl-features}

## Loading the data

The full dataset is stored in a single .csv file. The output class is the 9th column.

```clay
    // load the Pima dataset
    "Loading Pima dataset..." println
    data = Ann loadCsv("data/pima-indians-diabetes.csv")
    feature_count = features length
    "Data loaded with ${data cols} cols and ${data rows} rows" println
```

## Cleaning the data

As @fig-zeros below shows, there are a number of missing data entries. 

```{clay #fig-zeros caption="Number of 0 entries in each feature column."}
rows = data toArray
zeros = features map { |name| 0 }
rows each { |row|
    c = 0
    { c < feature_count } whileTrue {
        ((row at(c)) == 0) ifTrue { zeros setAt(c, (zeros at(c)) + 1) }
        c = c + 1
    }
}

Plot figure(#{figsize: [7, 3.5]})
bars = Plot bar(features, zeros, #{color: "steelblue"})
Plot barLabel(bars, #{fmt: "%d"})
Plot ylabel("Rows with 0")
Plot title("Zero entries by feature")
```

To analyze the data, we would either need to either drop those rows (losing precious data), or find some way to replace the missing data.

Several helper functions were created to find the mean
value of all training_data columns and to use that to replace missing data fields.

```clay
    // mean of the non-zero values in each column. Fitted on the training data only, so nothing
    // leaks in from the validation/test sets.
    imputeMeans = { |train_data|
        rows = train_data toArray
        means = []
        c = 0
        { c < train_data cols } whileTrue {
            sum = 0
            n = 0
            rows each { |row|
                v = row at(c)
                (v != 0) ifTrue { sum = sum + v; n = n + 1 }
            }
            means push((n > 0) ifTrue { sum / n } ifFalse { 0 })
            c = c + 1
        }
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

    // z-score the feature columns: (x - mean) / std. The label column is left untouched.
    standardize = { |data, means, stds|
        rows = data toArray
        rows each { |row|
            c = 0
            { c < feature_count } whileTrue {
                row setAt(c, ((row at(c)) - (means at(c))) / (stds at(c)))
                c = c + 1
            }
        }
        AnnData fromArray(rows)
    }

    // impute and standardize every split using statistics from the training split only
    clean = { |train_data, val_data, test_data|
        "Cleaning data (impute 0s, standardize)..." println
        means = imputeMeans(train_data)
        train_data = impute(train_data, means)
        val_data = impute(val_data, means)
        test_data = impute(test_data, means)

        mu = train_data mean
        sigma = train_data std
        [standardize(train_data, mu, sigma), standardize(val_data, mu, sigma), standardize(test_data, mu, sigma)]
    }
```

## Splitting the data

Now that we have the means to clean the data we can split the full dataset into training, validation and testing groups. We use an 80/10/10 split for this analysis.

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

# Model

Given the 8 input features and 1 output class, and a presumption that the problem was linear-logistic, a simple dense network model was chosen. Since this is a single class output predictor the loss function was set to `cross_entropy` which `libann` treats as `Binary Cross Entropy`.

Recall that the loss function for Binary Cross Entropy is:

$BCE = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1 - y_i)\log(1 - p_i)]$

## Model details

The following network properties were used to configure the network:

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

```clay
    // fraction of rows classified correctly, thresholding the sigmoid output at 0.5.
    // (net accuracy() argmaxes over output columns, which is always "class 0" with a single output.)
    accuracy = { |x, y|
        xs = x toArray
        ys = y toArray
        correct = 0
        i = 0
        { i < xs size } whileTrue {
            p = net predict(xs at(i))
            guess = ((p at(0)) >= 0.5) ifTrue { 1 } ifFalse { 0 }
            (guess == ((ys at(i)) at(0))) ifTrue { correct = correct + 1 }
            i = i + 1
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
    "Training accuracy:   ${accuracy(x_train, y_train) * 100}%" println
    "Validation accuracy: ${accuracy(x_val, y_val) * 100}%" println
    "Test accuracy:       ${accuracy(x_test, y_test) * 100}%" println

```

## Learning curve

As @fig-learn below shows, the training accuracy continues to show improvement. However, the validation accuracy levels out at around epoch 54. This is a clear indication of over-training and that the network is starting to memorize the training data.

Displaying the learning curve requires just a few lines of code.

```{clay #fig-learn caption="Learning curve."}
    history = net history
    Plot learningCurve(history)
    Plot title("Learning curve")
```
