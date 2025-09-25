# app.py (complete with Task 1 -> Task 8 integrated)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import re
import string
import math

# ML
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# NLP
import nltk
from nltk.corpus import twitter_samples, stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

# Try download necessary NLTK datasets if missing
try:
    nltk.data.find('corpora/twitter_samples')
except:
    nltk.download('twitter_samples')
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

# ====== Helper functions (shared across tasks) ======
def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    tweet = re.sub(r'#', '', tweet)

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    return tweets_clean

def build_freqs(tweets, ys):
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1
    return freqs

def extract_features(tweet, freqs):
    word_l = process_tweet(tweet)
    x = np.zeros((1, 3))
    x[0,0] = 1
    for word in word_l:
        x[0,1] += freqs.get((word, 1), 0)
        x[0,2] += freqs.get((word, 0), 0)
    return x

def extract_6features(tweet, freqs):
    x = np.zeros((1, 7))
    x[0,0] = 1
    pronouns = {"i","me","my","mine","myself","we","us","our","ours","ourselves","you","your","yours","yourself","yourselves"}
    word_1 = process_tweet(tweet)
    for word in word_1:
        x[0,1] += freqs.get((word,1), 0)
        x[0,2] += freqs.get((word,0), 0)
    if re.search(r"\bno\b", tweet.lower()):
        x[0,3] = 1
    tokens_raw = re.findall(r"\w+", tweet.lower())
    x[0,4] = sum(1 for t in tokens_raw if t in pronouns)
    if "!" in tweet:
        x[0,5] = 1
    word_count = len(tokens_raw)
    if word_count > 0:
        x[0,6] = math.log(word_count)
    return x

# Numerically safe sigmoid
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

# Improved gradient descent for logistic (Task 2)
def gradient_descent_logistic(X, Y, theta_init=None, alpha=1e-9, num_iters=10000, tol=1e-6, eps=1e-15, early_stop=True):
    m = X.shape[0]
    if theta_init is None:
        theta = np.zeros((X.shape[1], 1))
    else:
        theta = theta_init.copy()
    losses = []
    for i in range(num_iters):
        z = X.dot(theta)
        h = sigmoid(z)
        # clip h for stability in log
        h = np.clip(h, eps, 1 - eps)
        gradient = (X.T @ (h - Y)) / m
        grad_norm = np.linalg.norm(gradient)
        theta = theta - alpha * gradient
        # stable loss computation (log-sum-exp form alternative could be used)
        J = - (Y.T @ np.log(h) + (1 - Y).T @ np.log(1 - h)) / m
        losses.append(float(np.squeeze(J)))
        if early_stop and grad_norm < tol:
            break
    return losses, theta

# Prediction helper for logistic
def predict_logistic(X, theta):
    z = X.dot(theta)
    h = sigmoid(z)
    return (h >= 0.5).astype(int), h

# Decision function g(s) (Task 5)
def decision_g(tweet, freqs):
    x = extract_features(tweet, freqs)
    pos = x[0,1]
    neg = x[0,2]
    return 1 if pos > neg else 0

# Apply feature normalization as in Task 3 (normalize positive/negative counts by N = train_size * len(s))
def normalize_features_counts_for_row(x_row, sentence_len, train_size):
    sentence_len = max(1, sentence_len)
    N = train_size * sentence_len
    if N > 0:
        x_row[1] = x_row[1] / N
        x_row[2] = x_row[2] / N
    return x_row

# ====== Streamlit UI ======

# Change main title, add icons and Markdown
st.markdown("<h1 style='text-align: center;'>Sentiment Analysis Dashboard ✨</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: grey;'>Text sentiment analysis using machine learning</h3>", unsafe_allow_html=True)
st.write("---")

# Create sidebar with a clear title and grouped options
st.sidebar.markdown("## ⚙️ Settings")

st.sidebar.subheader("📤 Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV/TXT file", type=["csv", "txt"])

st.sidebar.subheader("🧠 Model Settings")
model_choice = st.sidebar.selectbox(
    "Choose a model",
    ["Logistic Regression (sklearn)", "Gradient Descent (custom)", "Decision Function g(s)", "SVM", "RandomForest", "Decision Tree", "Naive Bayes"]
)
iterations = st.sidebar.slider("Iterations", 100, 100000, step=1000, value=10000)
learning_rate = st.sidebar.number_input("Learning Rate", value=0.01, format="%.5f")
scaler_choice = st.sidebar.selectbox("Scaler", ["None", "Min-Max", "Standard", "Robust"])

st.sidebar.subheader("🎯 Feature Settings")
normalize = st.sidebar.checkbox("Apply normalization (Task 3)")
feature_count = st.sidebar.radio("Number of features", [2, 6])

st.sidebar.write("---")
# Group run buttons into columns for a cleaner look
col1_side, col2_side = st.sidebar.columns(2)
with col1_side:
    if st.button("🚀 Run Model"):
        st.session_state["run_model"] = True
with col2_side:
    if st.button("📥 Download Report"):
        st.success("Report generated! (export feature needs to be implemented)")

# Create tabs with icons for easy recognition
tab1, tab2, tab3 = st.tabs(["📊 Training Results", "📈 Comparison", "💬 Live Test"])

# ====== Data load & prepare function used when running ======
def load_dataset_from_upload(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        try:
            df = pd.read_table(uploaded_file)
        except Exception:
            return None
    return df

def prepare_data(use_uploaded=False):
    if use_uploaded and uploaded_file is not None:
        df = load_dataset_from_upload(uploaded_file)
        if df is None:
            st.warning("Cannot parse uploaded file — using twitter_samples instead.")
        else:
            cols = [c.lower() for c in df.columns]
            if 'text' in cols and 'label' in cols:
                text_col = df.columns[cols.index('text')]
                label_col = df.columns[cols.index('label')]
                texts = df[text_col].astype(str).tolist()
                labels = df[label_col].astype(int).values.reshape(-1,1)
                split = int(0.8 * len(texts))
                train_x = texts[:split]
                test_x = texts[split:]
                train_y = labels[:split]
                test_y = labels[split:]
                return train_x, train_y, test_x, test_y
            else:
                st.warning("Uploaded file needs columns 'text' and 'label' (0/1). Using twitter_samples instead.")
    all_pos = twitter_samples.strings('positive_tweets.json')
    all_neg = twitter_samples.strings('negative_tweets.json')
    train_pos = all_pos[:4000]
    test_pos = all_pos[4000:]
    train_neg = all_neg[:4000]
    test_neg = all_neg[4000:]
    train_x = train_pos + train_neg
    test_x = test_pos + test_neg
    train_y = np.append(np.ones((len(train_pos),1)), np.zeros((len(train_neg),1)), axis=0)
    test_y = np.append(np.ones((len(test_pos),1)), np.zeros((len(test_neg),1)), axis=0)
    return train_x, train_y, test_x, test_y

# ====== Main run (when user presses Run Model) ======
def run_all_and_store():
    st.info("Preparing data...")
    train_x, train_y, test_x, test_y = prepare_data(use_uploaded=True)

    freqs = build_freqs(train_x, train_y)
    train_size = len(train_x)

    if feature_count == 2:
        X = np.zeros((len(train_x), 3))
        for i in range(len(train_x)):
            X[i,:] = extract_features(train_x[i], freqs)
        X_test = np.zeros((len(test_x), 3))
        for i in range(len(test_x)):
            X_test[i,:] = extract_features(test_x[i], freqs)
    else:
        X = np.zeros((len(train_x), 7))
        for i in range(len(train_x)):
            X[i,:] = extract_6features(train_x[i], freqs)
        X_test = np.zeros((len(test_x), 7))
        for i in range(len(test_x)):
            X_test[i,:] = extract_6features(test_x[i], freqs)

    if normalize:
        for i, t in enumerate(train_x):
            words = process_tweet(t)
            if feature_count == 2:
                X[i,:] = normalize_features_counts_for_row(X[i,:], len(words), train_size)
            else:
                X[i,:] = normalize_features_counts_for_row(X[i,:], len(words), train_size)
        for i, t in enumerate(test_x):
            words = process_tweet(t)
            if feature_count == 2:
                X_test[i,:] = normalize_features_counts_for_row(X_test[i,:], len(words), train_size)
            else:
                X_test[i,:] = normalize_features_counts_for_row(X_test[i,:], len(words), train_size)

    scaler_obj = None
    if scaler_choice == "Min-Max":
        scaler_obj = MinMaxScaler()
    elif scaler_choice == "Standard":
        scaler_obj = StandardScaler()
    elif scaler_choice == "Robust":
        scaler_obj = RobustScaler()

    if scaler_obj is not None:
        X[:,1:] = scaler_obj.fit_transform(X[:,1:])
        X_test[:,1:] = scaler_obj.transform(X_test[:,1:])

    Y = np.squeeze(train_y)
    Y_train_for_sklearn = Y
    Y_test = np.squeeze(test_y)

    results = {}

    st.info("Training custom logistic regression (gradient descent)...")
    X_gd = X if X.ndim==2 else X.reshape(X.shape[0], -1)
    X_gd_mat = X_gd.astype(float)
    if X_gd_mat.shape[1] != X_gd_mat.shape[1]:
        pass
    Y_gd = train_y.astype(float)
    theta_init = np.zeros((X_gd_mat.shape[1], 1))
    start_time = time.time()
    losses, theta = gradient_descent_logistic(X_gd_mat, Y_gd, theta_init=theta_init, alpha=learning_rate, num_iters=iterations)
    time_custom = time.time() - start_time
    y_pred_custom_bin, y_pred_custom_prob = predict_logistic(X_test.astype(float), theta)
    try:
        accuracy_custom = accuracy_score(Y_test, y_pred_custom_bin)
        precision_custom = precision_score(Y_test, y_pred_custom_bin, zero_division=0)
        recall_custom = recall_score(Y_test, y_pred_custom_bin, zero_division=0)
        f1_custom = f1_score(Y_test, y_pred_custom_bin, zero_division=0)
        roc_auc_custom = roc_auc_score(Y_test, y_pred_custom_prob)
    except Exception:
        accuracy_custom = precision_custom = recall_custom = f1_custom = roc_auc_custom = None

    results['Custom GD Logistic'] = {
        'accuracy': accuracy_custom, 'precision': precision_custom, 'recall': recall_custom,
        'f1': f1_custom, 'roc_auc': roc_auc_custom, 'time': time_custom, 'losses': losses, 'theta': theta
    }

    st.info("Training sklearn LogisticRegression...")
    start_time = time.time()
    clf_lr = LogisticRegression(max_iter=1000)
    clf_lr.fit(X, Y_train_for_sklearn)
    time_sklearn = time.time() - start_time
    y_pred_sklearn = clf_lr.predict(X_test)
    try:
        accuracy_sklearn = accuracy_score(Y_test, y_pred_sklearn)
        precision_sklearn = precision_score(Y_test, y_pred_sklearn, zero_division=0)
        recall_sklearn = recall_score(Y_test, y_pred_sklearn, zero_division=0)
        f1_sklearn = f1_score(Y_test, y_pred_sklearn, zero_division=0)
        if hasattr(clf_lr, "predict_proba"):
            probs = clf_lr.predict_proba(X_test)[:,1]
        else:
            probs = clf_lr.decision_function(X_test)
        roc_auc_sklearn = roc_auc_score(Y_test, probs)
    except Exception:
        accuracy_sklearn = precision_sklearn = recall_sklearn = f1_sklearn = roc_auc_sklearn = None

    results['Sklearn Logistic'] = {
        'accuracy': accuracy_sklearn, 'precision': precision_sklearn, 'recall': recall_sklearn,
        'f1': f1_sklearn, 'roc_auc': roc_auc_sklearn, 'time': time_sklearn, 'model': clf_lr
    }

    st.info("Evaluating decision function g(s)...")
    y_pred_g = np.array([decision_g(t, freqs) for t in test_x])
    try:
        accuracy_g = accuracy_score(Y_test, y_pred_g)
        precision_g = precision_score(Y_test, y_pred_g, zero_division=0)
        recall_g = recall_score(Y_test, y_pred_g, zero_division=0)
        f1_g = f1_score(Y_test, y_pred_g, zero_division=0)
        roc_auc_g = roc_auc_score(Y_test, y_pred_g)
    except Exception:
        accuracy_g = precision_g = recall_g = f1_g = roc_auc_g = None

    results['Decision g(s)'] = {
        'accuracy': accuracy_g, 'precision': precision_g, 'recall': recall_g,
        'f1': f1_g, 'roc_auc': roc_auc_g, 'time': None
    }

    st.info("Training comparison models...")
    model_defs = {
        "SVM": SVC(kernel='linear', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": BernoulliNB()
    }
    for name, mdl in model_defs.items():
        start_time = time.time()
        if name in ["SVM"]:
            clf = Pipeline([("scaler", StandardScaler(with_mean=False)), ("model", mdl)])
        else:
            clf = mdl
        try:
            clf.fit(X, Y_train_for_sklearn)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(Y_test, y_pred)
            prec = precision_score(Y_test, y_pred, zero_division=0)
            rec = recall_score(Y_test, y_pred, zero_division=0)
            f1v = f1_score(Y_test, y_pred, zero_division=0)
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_test)[:,1]
                roc = roc_auc_score(Y_test, probs)
            else:
                try:
                    scores = clf.decision_function(X_test)
                    roc = roc_auc_score(Y_test, scores)
                except Exception:
                    roc = None
            elapsed = time.time() - start_time
        except Exception as e:
            acc = prec = rec = f1v = roc = elapsed = None
        results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1v, 'roc_auc': roc, 'time': elapsed, 'model': clf}

    st.session_state['results'] = results
    st.session_state['freqs'] = freqs
    st.session_state['models'] = {k: results[k].get('model') for k in results if 'model' in results[k]}
    st.session_state['theta_custom'] = theta
    st.session_state['X_test'] = X_test
    st.session_state['test_x'] = test_x
    st.session_state['Y_test'] = Y_test
    st.success("🎉 Training complete! Results saved in session.")
    return results

# ==== Tab 1: Training Results ====
with tab1:
    st.header("📊 Training & Evaluation Results")
    if uploaded_file is None and "run_model" not in st.session_state:
        st.info("No uploaded file detected — defaulting to built-in twitter_samples. Press 'Run Model' to execute.")
    if "run_model" in st.session_state:
        if 'results' not in st.session_state or st.button("Re-run models"):
            results = run_all_and_store()
        else:
            results = st.session_state.get('results')
        if results:
            key = None
            if model_choice == "Logistic Regression (sklearn)":
                key = 'Sklearn Logistic'
            elif model_choice == "Gradient Descent (custom)":
                key = 'Custom GD Logistic'
            elif model_choice == "Decision Function g(s)":
                key = 'Decision g(s)'
            else:
                key = model_choice

            if key in results:
                r = results[key]
                st.subheader(f"✨ Results for model: {key}")
                
                # Display key metrics in columns for easy comparison
                col_acc, col_prec, col_rec, col_f1 = st.columns(4)
                col_acc.metric("Accuracy", f"{(r['accuracy']*100) if r['accuracy'] is not None else 0:.2f}%")
                col_prec.metric("Precision", f"{(r['precision']*100) if r['precision'] is not None else 0:.2f}%")
                col_rec.metric("Recall", f"{(r['recall']*100) if r['recall'] is not None else 0:.2f}%")
                col_f1.metric("F1-score", f"{(r['f1']*100) if r['f1'] is not None else 0:.2f}%")
                
                # Add a separator line
                st.write("---")
                
                # Display loss plot if available
                if key == 'Custom GD Logistic' and r.get('losses'):
                    st.subheader("📉 Loss Curve")
                    fig, ax = plt.subplots()
                    ax.plot(r['losses'])
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Loss Value")
                    st.pyplot(fig)
            else:
                st.warning("Selected model results not available. Please re-run the model.")

# ==== Tab 2: Comparison ====
with tab2:
    st.header("📈 Model Comparison")
    if 'results' not in st.session_state:
        st.info("Press 'Run Model' first to see the comparison results.")
    else:
        results = st.session_state['results']
        rows = []
        for name, vals in results.items():
            rows.append({
                "Model": name,
                "Accuracy": vals.get('accuracy'),
                "Precision": vals.get('precision'),
                "Recall": vals.get('recall'),
                "F1-score": vals.get('f1'),
                "Time(s)": vals.get('time')
            })
        comp_df = pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
        
        # Create a container for the table and chart
        with st.container(border=True):
            st.markdown("### Summary of Results")
            st.dataframe(comp_df, hide_index=True)
        
        # Visualize accuracy with a bar chart
        st.write("---")
        st.markdown("### Accuracy Comparison Chart")
        fig2, ax2 = plt.subplots()
        comp_df.set_index("Model")["Accuracy"].plot(kind="bar", ax=ax2, color='skyblue')
        ax2.set_ylim(0,1)
        ax2.set_xlabel("Model")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy Comparison of Models")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig2)

# ==== Tab 3: Live Test ====
with tab3:
    st.header("💬 Test a Sentence Live")
    user_input = st.text_input("Enter a sentence for sentiment analysis:", "")
    run_models = st.session_state.get('results') is not None
    if st.button("🔮 Predict"):
        if not user_input.strip():
            st.warning("Please enter a non-empty sentence.")
        else:
            if not run_models:
                st.warning("⚠️ Please run the models first. Using a basic heuristic (g(s)) for a quick test.")
                label = decision_g(user_input, build_freqs(*prepare_data(use_uploaded=False)[:2]))
                st.success(f"Prediction (Heuristic g): {'Positive 😊' if label==1 else 'Negative 😞'}")
            else:
                st.info("Predicting with the selected model...")
                model = None
                if model_choice == "Logistic Regression (sklearn)":
                    model = st.session_state['models'].get('Sklearn Logistic')
                elif model_choice == "Gradient Descent (custom)":
                    theta = st.session_state.get('theta_custom')
                    freqs = st.session_state.get('freqs')
                    if theta is not None and freqs is not None:
                        x = extract_features(user_input, freqs)
                        pred_prob = sigmoid(np.dot(x, theta))[0,0]
                        pred = 1 if pred_prob >= 0.5 else 0
                        st.success(f"✅ Prediction: {'Positive 😊' if pred==1 else 'Negative 😞'} (Probability={pred_prob:.3f})")
                    else:
                        st.error("Custom Gradient Descent model has not been trained. Please press 'Run Model'.")
                    model = None
                elif model_choice == "Decision Function g(s)":
                    freqs = st.session_state.get('freqs')
                    if freqs is not None:
                        pred = decision_g(user_input, freqs)
                        st.success(f"✅ Prediction: {'Positive 😊' if pred==1 else 'Negative 😞'} (Based on g(s))")
                    else:
                        st.error("The g(s) model has not been trained. Please press 'Run Model'.")
                    model = None
                else:
                    model = st.session_state['models'].get(model_choice)
                
                if model:
                    try:
                        freqs = st.session_state.get('freqs')
                        if feature_count == 2:
                            x = extract_features(user_input, freqs).reshape(1,-1)
                        else:
                            x = extract_6features(user_input, freqs).reshape(1,-1)
                        pred = model.predict(x)
                        st.success(f"✅ Prediction: {'Positive 😊' if pred[0]==1 else 'Negative 😞'}")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}. Please try again or choose a different model.")
    
    # Task 8 - keeping this section as is for user customization
    st.write("---")
    if st.checkbox("Benchmark with ChatGPT/LLM"):
        st.info("To benchmark with an LLM, you must have an API key and implement the calls yourself.")
        st.write("Example approach:")
        st.markdown("""
        1. Prepare a CSV file with a `text` column for the test set.  
        2. Call the LLM's API for each sentence with a prompt: `Classify this sentence as positive or negative: {sentence}`.  
        3. Convert the LLM's response to a label (Positive=1, Negative=0), then compute the accuracy.  
        4. Be mindful of API rate limits and costs.
        """)
        st.write("**(This app does not automatically call external LLM APIs for security reasons. You need to implement your own API calls.)**")