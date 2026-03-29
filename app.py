from flask import Flask, request, jsonify, render_template, flash, redirect, url_for, send_file
import joblib
import numpy as np
import pandas as pd
import re
from collections import defaultdict
from scipy.special import expit
import os
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from io import BytesIO

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = "your_secret_key"

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

load_dotenv()

# ================= DATABASE =================

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ================= HELPERS =================

def is_admin():
    return current_user.is_authenticated and current_user.role == "admin"

# ================= MODELS =================

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    text = db.Column(db.Text, nullable=False)
    label = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    model_used = db.Column(db.String(100))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class ProductReview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    product_name = db.Column(db.String(200))
    review_text = db.Column(db.Text, nullable=False)
    label = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    explanation = db.Column(db.JSON)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User")  # 🔥 IMPORTANT

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    email = db.Column(db.String(150), unique=True)
    password_hash = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    role = db.Column(db.String(50), default="user")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ================= LOAD MODEL =================

saved = joblib.load("trustlens_chicago_model.pkl")

lr_model = saved["lr"]
svm_model = saved["svm"]
rf_model = saved["rf"]
xgb_model = saved["xgb"]
vectorizer = saved["vectorizer"]

# ================= ANALYTICS =================

analytics = {
    "total_predictions": 0,
    "spam_count": 0,
    "genuine_count": 0,
    "model_usage": defaultdict(int),
    "recent_predictions": []
}

# ================= UTIL =================

def clean_text(text):
    return re.sub(r"[^a-zA-Z\s]", "", text.lower()).strip()

# ================= MODEL =================

def run_model(text, model, model_name):

    features = vectorizer.transform([clean_text(text)])
    prediction = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(features)[0]))
    else:
        confidence = float(expit(abs(model.decision_function(features)[0])))

    label = "Deceptive (Fake Review)" if prediction == 1 else "Truthful (Genuine Review)"

    return label, confidence, {}

def predict_review(text, model_choice):

    model_map = {
        "lr": (lr_model, "Logistic Regression"),
        "svm": (svm_model, "Linear SVM"),
        "rf": (rf_model, "Random Forest"),
        "xgb": (xgb_model, "XGBoost")
    }

    model, name = model_map.get(model_choice, (svm_model, "Linear SVM"))

    label, confidence, explanation = run_model(text, model, name)

    analytics["total_predictions"] += 1
    analytics["model_usage"][name] += 1

    if "Deceptive" in label:
        analytics["spam_count"] += 1
    else:
        analytics["genuine_count"] += 1

    return label, confidence, name, explanation

# ================= ROUTES =================

# 👉 HOME (WELCOME PAGE)
@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("review_checker"))
    return render_template("index.html")  # welcome handled in template


# 👉 REVIEW CHECK PAGE
@app.route("/review-checking")
@login_required
def review_checker():
    return render_template("index.html")


# 👉 PRODUCT PAGE (ALL USERS SEE ALL REVIEWS)
@app.route("/product")
@login_required
def product_page():
    reviews = ProductReview.query.order_by(ProductReview.timestamp.desc()).all()
    return render_template("product.html", reviews=reviews)


# 👉 USER DASHBOARD
# @app.route("/dashboard")
# @login_required
# def dashboard():
#     if is_admin():
#         return redirect(url_for("admin"))
#     return render_template("dashboard.html")
@app.route("/dashboard")
@login_required
def dashboard():

    if is_admin():
        return redirect(url_for("admin"))

    user_reviews = Review.query.filter_by(user_id=current_user.id)

    total = user_reviews.count()
    fake = user_reviews.filter(Review.label.contains("Deceptive")).count()
    genuine = user_reviews.filter(Review.label.contains("Truthful")).count()

    return render_template(
        "dashboard.html",
        total=total,
        fake=fake,
        genuine=genuine
    )


# 👉 ADMIN DASHBOARD
@app.route("/admin")
@login_required
def admin():
    if not is_admin():
        return redirect(url_for("dashboard"))
    return render_template("admin.html")


# 👉 MODEL COMPARISON (ADMIN)
@app.route("/comparison")
@login_required
def comparison():
    if not is_admin():
        return redirect(url_for("dashboard"))
    return render_template("comparison.html")


# 👉 BATCH (ADMIN)
@app.route("/batch")
@login_required
def batch():
    if not is_admin():
        return redirect(url_for("dashboard"))
    return render_template("batch.html")


# ================= AUTH =================


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":

        username = request.form["username"]
        email = request.form["email"].lower()
        password = generate_password_hash(request.form["password"])

        if User.query.filter((User.email == email) | (User.username == username)).first():
            return render_template("register.html", error="User already exists")

                # Auto-assign admin based on email
        role = "admin" if email == "admin@gmail.com" else "user"

        user = User(
            username=username,
            email=email,
            password_hash=password,
            role=role
        )
        
        db.session.add(user)
        db.session.commit()


        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":

        email = request.form["email"].lower()
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, request.form["password"]):
            login_user(user)

            return redirect(url_for("admin" if user.role == "admin" else "review_checker"))

        return render_template("login.html", error="Invalid Credentials")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ================= CORE =================

@app.route("/predict", methods=["POST"])
@login_required
def predict():

    data = request.json
    text = data.get("review_text")

    label, confidence, model_name, _ = predict_review(text, data.get("model"))

    db.session.add(Review(
        text=text,
        label=label,
        confidence=round(confidence * 100, 2),
        model_used=model_name,
        user_id=current_user.id
    ))

    db.session.commit()

    return jsonify({
        "prediction": label,
        "confidence": round(confidence * 100, 2),
        "model": model_name
    })


@app.route("/submit_review", methods=["POST"])
@login_required
def submit_review():

    data = request.json
    text = data.get("review_text")

    label, confidence, _, explanation = predict_review(text, "svm")

    db.session.add(ProductReview(
        review_text=text,
        label=label,
        confidence=round(confidence * 100, 2),
        explanation=explanation,
        user_id=current_user.id
    ))

    db.session.commit()

    return jsonify({
        "review": text,
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    })


# ================= ANALYTICS =================

@app.route("/user_dashboard")
@login_required
def user_dashboard():
    reviews = Review.query.filter_by(user_id=current_user.id).order_by(Review.timestamp.desc()).all()
    return render_template("user_dashboard.html", reviews=reviews)

# @app.route("/analytics")
# @login_required
# def analytics_data():

#     users = User.query.all()

#     user_data = [
#         {
#             "username": u.username,
#             "reviews_count": Review.query.filter_by(user_id=u.id).count()
#         } for u in users
#     ]

#     return jsonify({
#         "total_predictions": Review.query.count(),
#         "spam_count": Review.query.filter(Review.label.contains("Deceptive")).count(),
#         "genuine_count": Review.query.filter(Review.label.contains("Truthful")).count(),
#         "model_usage": dict(analytics["model_usage"]),
#         "users": user_data
#     })

@app.route("/analytics")
@login_required
def analytics_data():

    if is_admin():
        # ADMIN → full system data
        return jsonify({
            "total_predictions": Review.query.count(),
            "spam_count": Review.query.filter(Review.label.contains("Deceptive")).count(),
            "model_usage": dict(analytics["model_usage"]),
            "genuine_count": Review.query.filter(Review.label.contains("Truthful")).count()
        })

    # USER → only their data
    user_reviews = Review.query.filter_by(user_id=current_user.id)

    return jsonify({
        "total_predictions": user_reviews.count(),
        "spam_count": user_reviews.filter(Review.label.contains("Deceptive")).count(),
        "model_usage": {},
        "genuine_count": user_reviews.filter(Review.label.contains("Truthful")).count()
    })


@app.route("/admin_dashboard")
@login_required
def admin_dashboard():

    if not is_admin():
        return jsonify({"error": "Access denied"}), 403

    users = User.query.all()

    return jsonify([
        {
            "username": u.username,
            "reviews_count": Review.query.filter_by(user_id=u.id).count()
        } for u in users
    ])




@app.route("/batch_predict", methods=["POST"])
@login_required
def batch_predict():

    if not is_admin():
        return jsonify({"error": "Access denied"}), 403

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        df = pd.read_csv(file)

        if "review" not in df.columns:
            return jsonify({"error": "CSV must contain 'review' column"}), 400

        results = []

        for text in df["review"]:
            label, confidence, model_name, _ = predict_review(text, "svm")

            results.append({
                "review": text[:150],
                "prediction": label,
                "confidence": round(confidence * 100, 2)
            })

        return jsonify({
            "total_reviews": len(results),
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================= RUN =================

if __name__ == "__main__":
   with app.app_context():
    db.create_all()

    # Create admin if not exists
    admin = User.query.filter_by(email="admin@gmail.com").first()
    
    if not admin:
        admin = User(
            username="admin",
            email="admin@gmail.com",
            password_hash=generate_password_hash("admin123"),
            role="admin"
        )
        db.session.add(admin)
        db.session.commit()
        print("✅ Admin auto-created")
    app.run(debug=True)