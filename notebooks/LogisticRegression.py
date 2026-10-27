
import pandas as pd

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATA
# ==========================================

data = pd.read_parquet("data/may_modeling_data.parquet")

print("Rows:", len(data))

print("\nFraud labels:")
print(data["is_fraud_event"].value_counts())

X = data.drop(columns=["is_fraud_event"])
y = data["is_fraud_event"]


# ==========================================
# 2. DEFINE FEATURES
# ==========================================

# Original Model 1: 13 features
model1_features = [
    "booking_lead_days",
    "stay_length",
    "same_day_booking",
    "rate_code",
    "booking_channel",
    "member_tenure_days",
    "total_reservations",
    "single_night_ratio",
    "same_day_booking_count",
    "qns_count",
    "qxy_count",
    "qns_qxy_rate",
    "qns_qxy_points_earned"
]

# Conservative Model 1: remove four historical
# qualification/points features with uncertain timing
excluded_features = [
    "qns_count",
    "qxy_count",
    "qns_qxy_rate",
    "qns_qxy_points_earned"
]

conservative_features = [
    feature
    for feature in model1_features
    if feature not in excluded_features
]

# Model 2: 18 features
model2_features = model1_features + [
    "qualification_code",
    "current_qns",
    "current_qxy",
    "points_earned",
    "room_revenue"
]


# ==========================================
# 3. ORIGINAL RANDOM SPLIT DIAGNOSTIC
# ==========================================

random_X_train, random_X_test, random_y_train, random_y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
)

train_members = set(
    random_X_train["member_number"].dropna()
)

test_members = set(
    random_X_test["member_number"].dropna()
)

print("\nRANDOM SPLIT CHECK")
print("Training rows:", len(random_X_train))
print("Testing rows:", len(random_X_test))
print("Members in both:", len(train_members & test_members))

train_fraud_members = set(
    random_X_train.loc[
        random_y_train == 1, "member_number"
    ].dropna()
)

test_fraud_members = set(
    random_X_test.loc[
        random_y_test == 1, "member_number"
    ].dropna()
)

print(
    "Fraud members in both:",
    len(train_fraud_members & test_fraud_members)
)


# ==========================================
# 4. MEMBER-BASED TRAIN/TEST SPLIT
# ==========================================

groups = X["member_number"].copy()

# Give nonmember reservations separate groups
groups = groups.fillna(
    pd.Series(
        "nonmember_" + X.index.astype(str),
        index=X.index
    )
)

group_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    group_split.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

group_train_members = set(
    X_train["member_number"].dropna()
)

group_test_members = set(
    X_test["member_number"].dropna()
)

print("\nGROUP SPLIT CHECK")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print(
    "Member overlap:",
    len(group_train_members & group_test_members)
)

print("\nTraining fraud labels:")
print(y_train.value_counts())

print("\nTesting fraud labels:")
print(y_test.value_counts())


# ==========================================
# 5. SIMPLE TWO-FEATURE FRAUD RULE
# ==========================================

# Rule discovered during exploratory May analysis:
# booking lead <= 3 days AND member tenure <= 28 days.
#
# IMPORTANT:
# These cutoffs were identified using all May labels,
# including test labels. This is a diagnostic comparison,
# NOT an independent validation of the rule.

simple_rule_predictions = (
    (X_test["booking_lead_days"] <= 3) &
    (X_test["member_tenure_days"] <= 28)
).astype(int)

simple_rule_precision = precision_score(
    y_test, simple_rule_predictions, zero_division=0
)

simple_rule_recall = recall_score(
    y_test, simple_rule_predictions, zero_division=0
)

simple_rule_f1 = f1_score(
    y_test, simple_rule_predictions, zero_division=0
)

simple_tn, simple_fp, simple_fn, simple_tp = confusion_matrix(
    y_test,
    simple_rule_predictions,
    labels=[0, 1]
).ravel()

simple_fpr = (
    simple_fp / (simple_fp + simple_tn)
    if (simple_fp + simple_tn) > 0
    else 0
)

print("\nSIMPLE RULE - HELD-OUT MEMBERS")
print("Precision:", simple_rule_precision)
print("Recall:", simple_rule_recall)
print("F1:", simple_rule_f1)
print("False Positive Rate:", simple_fpr)
print("True Negatives:", simple_tn)
print("False Positives:", simple_fp)
print("False Negatives:", simple_fn)
print("True Positives:", simple_tp)


# ==========================================
# 6. SELECT MODEL FEATURES
# ==========================================

X1_train = X_train[model1_features]
X1_test = X_test[model1_features]

X_conservative_train = X_train[conservative_features]
X_conservative_test = X_test[conservative_features]

X2_train = X_train[model2_features]
X2_test = X_test[model2_features]


# ==========================================
# 7. PREPROCESSING
# ==========================================

def make_numeric_transformer():
    return Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value=0,
            add_indicator=True
        )),
        ("scaler", StandardScaler())
    ])


def make_categorical_transformer():
    return Pipeline([
        ("onehot", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])


def build_model(features):

    categorical_features = [
        feature
        for feature in features
        if feature in [
            "rate_code",
            "booking_channel",
            "qualification_code"
        ]
    ]

    numeric_features = [
        feature
        for feature in features
        if feature not in categorical_features
    ]

    preprocessor = ColumnTransformer([
        (
            "num",
            make_numeric_transformer(),
            numeric_features
        ),
        (
            "cat",
            make_categorical_transformer(),
            categorical_features
        )
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000
        ))
    ])


# ==========================================
# 8. CREATE AND TRAIN MODELS
# ==========================================

model1 = build_model(model1_features)

conservative_model = build_model(conservative_features)

model2 = build_model(model2_features)

print("\nTraining Model 1...")
model1.fit(X1_train, y_train)

print("\nTraining Conservative Model 1...")
conservative_model.fit(
    X_conservative_train,
    y_train
)

print("\nTraining Model 2...")
model2.fit(X2_train, y_train)


# ==========================================
# 9. MODEL EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    pr_auc = average_precision_score(
        y_test, scores
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\n{name} RESULTS")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("PR-AUC:", pr_auc)
    print("False Positive Rate:", fpr)

    print("\nConfusion Matrix:")
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)

    return {
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "PR-AUC": pr_auc,
        "FPR": fpr,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp
    }


# ==========================================
# 10. EVALUATE ALL THREE MODELS
# ==========================================

results = []

results.append(
    evaluate_model(
        "Model 1 (Original)",
        model1,
        X1_test,
        y_test
    )
)

results.append(
    evaluate_model(
        "Model 1 (Conservative)",
        conservative_model,
        X_conservative_test,
        y_test
    )
)

results.append(
    evaluate_model(
        "Model 2 (Original)",
        model2,
        X2_test,
        y_test
    )
)


# ==========================================
# 11. MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame(results)

# Add the simple rule to the comparison.
# PR-AUC is omitted because this rule produces
# binary predictions rather than probability scores.

simple_rule_result = pd.DataFrame([{
    "Model": "Simple Two-Feature Rule",
    "Precision": simple_rule_precision,
    "Recall": simple_rule_recall,
    "F1": simple_rule_f1,
    "PR-AUC": float("nan"),
    "FPR": simple_fpr,
    "False Positives": simple_fp,
    "False Negatives": simple_fn,
    "True Positives": simple_tp
}])

comparison = pd.concat(
    [comparison, simple_rule_result],
    ignore_index=True
)

print("\nMODEL COMPARISON (SAME TEST MEMBERS)")
print(comparison.to_string(index=False))


# ==========================================
# 12. FEATURE COEFFICIENTS
# ==========================================

def print_coefficients(name, model):

    feature_names = (
        model.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    coefficients = (
        model.named_steps["classifier"].coef_[0]
    )

    weights = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients
    })

    weights = weights.sort_values(
        "coefficient",
        ascending=False
    )

    print(f"\n{name} FEATURE COEFFICIENTS")
    print(weights.to_string(index=False))


print_coefficients("MODEL 1", model1)

print_coefficients(
    "CONSERVATIVE MODEL 1",
    conservative_model
)

print_coefficients("MODEL 2", model2)
