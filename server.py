from flask import Flask, request, jsonify
from medical_converter import simplify_medical_text

app = Flask(__name__)

@app.route('/simplify', methods=['POST'])
def simplify():
    data = request.get_json()
    medical_input = data.get("text", "")

    if not medical_input.strip():
        return jsonify({"error": "Empty input"}), 400

    try:
        output = simplify_medical_text(medical_input)
        return jsonify({"layman": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
