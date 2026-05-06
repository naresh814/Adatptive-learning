"""
Authentication Blueprint — ASD Adaptive Learning System
=========================================================
Provides:
  - User model (SQLAlchemy) with bcrypt password hashing
  - POST /register  → create new user
  - POST /login     → validate credentials, set session
  - GET  /logout    → clear session, redirect to login
  - GET  /me        → return current user info (JSON)
  - login_required  → decorator to protect routes
"""

from functools import wraps
from flask import (
    Blueprint, request, jsonify, session, redirect, url_for
)

# These will be initialised by init_auth() called from app.py
db     = None
bcrypt = None

auth_bp = Blueprint("auth", __name__)


# ── User Model ────────────────────────────────────────────────────────────────

def _get_user_model():
    """Returns the User model class.  Defined as a function so the model
    is created *after* db has been set by init_auth()."""

    class User(db.Model):
        __tablename__ = "users"
        id       = db.Column(db.Integer, primary_key=True)
        username = db.Column(db.String(80), unique=True, nullable=False)
        password = db.Column(db.String(200), nullable=False)   # bcrypt hash

        def __repr__(self):
            return f"<User {self.username}>"

    return User


User = None  # populated by init_auth()


def init_auth(app, _db, _bcrypt):
    """Call this from app.py after creating db & bcrypt instances."""
    global db, bcrypt, User
    db     = _db
    bcrypt = _bcrypt
    User   = _get_user_model()
    app.register_blueprint(auth_bp)
    return User


# ── Decorator ─────────────────────────────────────────────────────────────────

def login_required(f):
    """Protect a route — redirects browsers to /login, returns 401 for API/fetch."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            # Detect fetch/XHR/API calls:
            #   - X-Requested-With header (jQuery AJAX, etc.)
            #   - Accept header asking for JSON
            #   - Non-navigational requests (POST, PUT, DELETE)
            is_api = (
                request.headers.get("X-Requested-With", "").lower() == "xmlhttprequest"
                or "application/json" in request.headers.get("Accept", "")
                or request.is_json
                or request.method in ("POST", "PUT", "DELETE")
            )
            if is_api:
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("auth.login_page"))
        return f(*args, **kwargs)
    return decorated


# ── Page Routes ───────────────────────────────────────────────────────────────

@auth_bp.route("/login")
def login_page():
    """Serve the login page.  If already logged in, go to dashboard."""
    if "user_id" in session:
        return redirect("/")
    from flask import render_template
    return render_template("login.html")


@auth_bp.route("/register_page")
def register_page():
    """Serve the register page."""
    if "user_id" in session:
        return redirect("/")
    from flask import render_template
    return render_template("register.html")


# ── API Routes ────────────────────────────────────────────────────────────────

@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Invalid JSON body"}), 400

    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    # ── Validation ────────────────────────────────────────
    if not username or not password:
        return jsonify({"success": False, "error": "Username and password are required"}), 400

    if len(username) < 3:
        return jsonify({"success": False, "error": "Username must be at least 3 characters"}), 400

    if len(password) < 6:
        return jsonify({"success": False, "error": "Password must be at least 6 characters"}), 400

    # ── Check if username already taken ───────────────────
    existing = User.query.filter_by(username=username).first()
    if existing:
        return jsonify({"success": False, "error": "Username already taken"}), 409

    # ── Create user ───────────────────────────────────────
    hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
    new_user  = User(username=username, password=hashed_pw)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"success": True, "message": f"User '{username}' registered successfully"}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Invalid JSON body"}), 400

    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"success": False, "error": "Username and password are required"}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not bcrypt.check_password_hash(user.password, password):
        return jsonify({"success": False, "error": "Invalid username or password"}), 401

    # ── Set session ───────────────────────────────────────
    session["user_id"]  = user.id
    session["username"] = user.username

    return jsonify({
        "success":  True,
        "message":  "Login successful",
        "username": user.username,
    })


@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login_page"))


@auth_bp.route("/me")
def me():
    """Return current logged-in user info, or 401."""
    if "user_id" not in session:
        return jsonify({"logged_in": False}), 401
    return jsonify({
        "logged_in": True,
        "user_id":   session["user_id"],
        "username":  session["username"],
    })
