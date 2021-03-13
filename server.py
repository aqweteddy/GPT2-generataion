import time
from flask import Flask, jsonify, request
from argparse import ArgumentParser
from generate import Generator


parser = ArgumentParser()
parser.add_argument('--model_type', type=str, default='gpt2-calibration')
parser.add_argument('--model_dir', type=str, default='gpt2_ckpt/epoch=2')
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()

generator = Generator(args.model_dir, args.model_type, args.device)

app = Flask(__name__)


@app.route('/predict')
def predict():
    start = time.time()
    text: str = request.args.get('text')
    minlen: int = int(request.args.get('minlen', 50))
    maxlen: int = int(request.args.get('maxlen', 150))
    numseq: int = int(request.args.get('numseq', 5))
    temperature: float = float(request.args.get('temperature', 1.5))
    result = generator.generate(text, min_length=minlen,
                                max_length=maxlen,
                                num_seq=numseq,
                                temperature=temperature)
    return jsonify({'result': result, 'time': time.time() - start})


app.run(debug=True)