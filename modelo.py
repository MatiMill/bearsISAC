from limpieza import cargar_datos
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss

df = cargar_datos()
print(df.shape)

df = df[df["is_swing"] == 1].copy()
print(len(df), df["is_whiff"].mean())

features = ["RelSpeed", "SpinRate", "InducedVertBreak", "HorzBreak",
            "Extension", "RelHeight", "RelSide"]

#Lo datos tomados en consideracion
X = df[features]
#Lo que se intenta predecir
y = df["is_whiff"]

#entrena en 80% y se testea con 20% de los datos
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#evita entrenarse con datos de pitchers ya conocidos
train_idx, test_idx = next(splitter.split(X, y, groups=df["pitcher_anon_id"]))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#200 decision trees, cuanto vale cada arbol, cada regla aprendida minimo en 100 casos elimina ruido
modelo = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, min_child_samples=100, verbose=-1)
modelo.fit(X_train, y_train)

#get prediction, outputs prob contact prob miss, keeps miss probs
p_test = modelo.predict_proba(X_test)[:, 1]
#asks if ranked correctly, compares to real scores, how much better than chance

print("How often is the model correct:")
print("AUC", roc_auc_score(y_test, p_test))

print("Average penalty of wrong guess, lower is better... :")
print("Log loss model:   ", log_loss(y_test, p_test))
print("Log loss baseline:", log_loss(y_test, [y_train.mean()] * len(y_test)))

#Stuff+ metric creation
test = df.iloc[test_idx].copy()
test["p_whiff"] = p_test
test["stuff_plus"] = 100 + 10 * (test["p_whiff"] - test["p_whiff"].mean()) / test["p_whiff"].std()

print(test.groupby("AutoPitchType")["stuff_plus"].mean().round(1).sort_values(ascending=False))

#best pitchers prediction
por_pitcher = test.groupby("pitcher_anon_id").filter(lambda g: len(g) >= 100)
print(por_pitcher.groupby("pitcher_anon_id")["stuff_plus"].mean().round(1).sort_values(ascending=False).head(10))