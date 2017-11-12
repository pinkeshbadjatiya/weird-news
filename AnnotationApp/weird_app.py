from flask import Flask, render_template, request
import random, copy
import csv
import pdb
import json

app = Flask(__name__)

final_answers = {}
temp=['0','1','2','3']
read_count = 500

def load_ques(annotator_no):
    with open("weirdnews_annotated" + str(annotator_no) + ".json") as f:
        data = f.readlines()

    data = data[:read_count]
    ques = {}
    for i, row in enumerate(data):
        jsn = json.loads(row)
        ques[i+1] = {
                'text': jsn['text'].strip(),
                'annotation': -1            # Fill this after the annotation
                }
        if 'annotation' in jsn:
            ques[i+1]['annotation'] = jsn['annotation']

    return ques


def dump_to_file(annotator_no, questions):
    # Save
    data = []
    for qno in range(1, read_count+1):
        data += [json.dumps(questions[qno]) + "\n"]
    with open('weirdnews_annotated' + str(annotator_no) + '.json', 'wb') as f:
        f.writelines(data)


questions1 = load_ques(1)
questions2 = load_ques(2)
questions3 = load_ques(3)


@app.route('/annotate1')
def quiz1():
    remaining = len([k for k in questions1 if questions1[k]['annotation'] == -1])
    print "\tAnnotator1: Loading page, require", str(remaining), "annotations"
    return render_template('main.html', ques=questions1, total=read_count, savepath="/save1", remaining=remaining)

@app.route('/annotate2')
def quiz2():
    remaining = len([k for k in questions2 if questions2[k]['annotation'] == -1])
    print "\tAnnotator2: Loading page, require", str(remaining), "annotations"
    return render_template('main.html', ques=questions2, total=read_count, savepath="/save2", remaining=remaining)

@app.route('/annotate3')
def quiz3():
    remaining = len([k for k in questions3 if questions3[k]['annotation'] == -1])
    print "\tAnnotator3: Loading page, require", str(remaining), "annotations"
    return render_template('main.html', ques=questions3, total=read_count, savepath="/save3", remaining=remaining)


@app.route('/save1', methods=['POST'])
def quiz_answers1():
    correct = 0
    # qno is 1 indexed
    for qno in request.form:
        questions1[int(qno)]['annotation'] = int(request.form[qno])
        print "\tAnnotator1: Saving annotation for", qno
    dump_to_file(1, questions1)
    return '<h1>Thankyou 1</h1>'


@app.route('/save2', methods=['POST'])
def quiz_answers2():
    correct = 0
    # qno is 1 indexed
    for qno in request.form:
        questions2[int(qno)]['annotation'] = int(request.form[qno])
        print "\tAnnotator2: Saving annotation for", qno
    dump_to_file(2, questions2)
    return '<h1>Thankyou 2</h1>'

@app.route('/save3', methods=['POST'])
def quiz_answers3():
    correct = 0
    # qno is 1 indexed
    for qno in request.form:
        questions3[int(qno)]['annotation'] = int(request.form[qno])
        print "\tAnnotator3: Saving annotation for", qno
    dump_to_file(3, questions3)
    return '<h1>Thankyou 3</h1>'


if __name__ == '__main__':
    app.run(port="8080", host="0.0.0.0")
