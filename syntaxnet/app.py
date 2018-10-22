from flask import Flask, request
import json
import os
import ipywidgets as widgets
import tensorflow as tf
from IPython import display
from dragnn.protos import spec_pb2
from dragnn.python import graph_builder
from dragnn.python import spec_builder
from dragnn.python import load_dragnn_cc_impl  # This loads the actual op definitions
from dragnn.python import render_parse_tree_graphviz
from dragnn.python import visualization
from google.protobuf import text_format
from syntaxnet import load_parser_ops  # This loads the actual op definitions
from syntaxnet import sentence_pb2
from syntaxnet.ops import gen_parser_ops
from tensorflow.python.platform import tf_logging as logging

class SyntaxParser():

    def __init__(self):
        self.segmenter_model = self.load_model("English/segmenter", "spec.textproto", "checkpoint")
        self.parser_model = self.load_model("English", "parser_spec.textproto", "checkpoint")


    def load_model(self, base_dir, master_spec_name, checkpoint_name):
        # Read the master spec
        master_spec = spec_pb2.MasterSpec()
        with open(os.path.join(base_dir, master_spec_name), "r") as f:
            text_format.Merge(f.read(), master_spec)
        spec_builder.complete_master_spec(master_spec, None, base_dir)
        logging.set_verbosity(logging.WARN)  # Turn off TensorFlow spam.

        # Initialize a graph
        graph = tf.Graph()
        with graph.as_default():
            hyperparam_config = spec_pb2.GridPoint()
            builder = graph_builder.MasterBuilder(master_spec, hyperparam_config)
            # This is the component that will annotate test sentences.
            annotator = builder.add_annotation(enable_tracing=True)
            builder.add_saver()  # "Savers" can save and load models; here, we're only going to load.

        sess = tf.Session(graph=graph)
        with graph.as_default():
            # sess.run(tf.global_variables_initializer())
            # sess.run('save/restore_all', {'save/Const:0': os.path.join(base_dir, checkpoint_name)})
            builder.saver.restore(sess, os.path.join(base_dir, checkpoint_name))

        def annotate_sentence(sentence):
            with graph.as_default():
                return sess.run([annotator['annotations'], annotator['traces']],
                                feed_dict={annotator['input_batch']: [sentence]})

        return annotate_sentence

    def annotate_text(self, text):
        sentence = sentence_pb2.Sentence(
            text=text,
            token=[sentence_pb2.Token(word=text, start=-1, end=-1)]
        )

        # preprocess
        with tf.Session(graph=tf.Graph()) as tmp_session:
            char_input = gen_parser_ops.char_token_generator([sentence.SerializeToString()])
            preprocessed = tmp_session.run(char_input)[0]
        segmented, _ = self.segmenter_model(preprocessed)

        annotations, traces = self.parser_model(segmented[0])
        assert len(annotations) == 1
        assert len(traces) == 1
        return sentence_pb2.Sentence.FromString(annotations[0]), traces[0]

app = Flask(__name__)
parser = SyntaxParser()


@app.route("/", methods=['GET'])
def hello():
    return "Hello World!"

@app.route("/parse", methods=['POST'])
def parse():
    json_chunk = request.get_json()
    result = []
    if 'sentence' in json_chunk:
        annotated_text = parser.annotate_text(json_chunk['sentence'])
        for i, tok in enumerate(annotated_text[0].token):
            parent = -10
            if tok.head:
                parent = tok.head
            pos = ''
            split_tag = tok.tag.split('++')
            if len(split_tag) > 1:
                pos = split_tag[1].split('\"')[0]
            result.append({'id' : i, 'word' : tok.word.lower(), 'dependency_label' : tok.label, 'pos' : pos, 'parent' : parent})

    response = app.response_class(
        response = json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, threaded=True)