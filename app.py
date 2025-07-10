
from flask import Flask, request, render_template, jsonify
from search_engine import search_and_summarize

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "Empty query"}), 400
    try:
        answer, sources = search_and_summarize(user_query)
        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
