import flask
import nltk
nltk.download('all')
from Preprocess import Preprocess
from flask import request, jsonify
from  flask_cors import CORS
preprocessing = Preprocess()

tfidf_vectorizer=preprocessing.readFile("./tfidf.pickle")
clf = preprocessing.readFile("./clf.pickle")

app = flask.Flask(__name__)
cors = CORS(app)
app.config["DEBUG"] = True



@app.route('/test', methods=['GET'])
def home():

    news="UNITED NATIONS (Reuters) - Two North Korean shipments to a Syrian government agency responsible for the country s chemical weapons program were intercepted in the past six months, according to a confidential United Nations report on North Korea sanctions violations. The report by a panel of independent U.N. experts, which was submitted to the U.N. Security Council earlier this month and seen by Reuters on Monday, gave no details on when or where the interdictions occurred or what the shipments contained.   The panel is investigating reported prohibited chemical, ballistic missile and conventional arms cooperation between Syria and the DPRK (North Korea),  the experts wrote in the 37-page report.   Two member states interdicted shipments destined for Syria. Another Member state informed the panel that it had reasons to believe that the goods were part of a KOMID contract with Syria,  according to the report. KOMID is the Korea Mining Development Trading Corporation. It was blacklisted by the Security Council in 2009 and described as Pyongyang s key arms dealer and exporter of equipment related to ballistic missiles and conventional weapons. In March 2016 the council also blacklisted two KOMID representatives in Syria.   The consignees were Syrian entities designated by the European Union and the United States as front companies for Syria s Scientific Studies and Research Centre (SSRC), a Syrian entity identified by the Panel as cooperating with KOMID in previous prohibited item transfers,  the U.N. experts wrote.  SSRC has overseen the country s chemical weapons program since the 1970s. The U.N. experts said activities between Syria and North Korea they were investigating included cooperation on Syrian Scud missile programs and maintenance and repair of Syrian surface-to-air missiles air defense systems. The North Korean and Syrian missions to the United Nations did not immediately respond to a request for comment.  The experts said they were also investigating the use of the VX nerve agent in Malaysia to kill the estranged half-brother of North Korea s leader Kim Jong Un in February.  North Korea has been under U.N. sanctions since 2006 over its ballistic missile and nuclear programs and the Security Council has ratcheted up the measures in response to five nuclear weapons tests and four long-range missile launches. Syria agreed to destroy its chemical weapons in 2013 under a deal brokered by Russia and the United States. However, diplomats and weapons inspectors suspect Syria may have secretly maintained or developed a new chemical weapons capability. During the country s more than six-year long civil war the Organisation for the Prohibition of Chemical Weapons has said the banned nerve agent sarin has been used at least twice, while the use of chlorine as a weapon has been widespread. The Syrian government has repeatedly denied using chemical weapons. "

    sample=[news]
    sample=tfidf_vectorizer.transform(sample).toarray()

    print(sample)
    pred=clf.predict(sample)

    return jsonify({"pred":int(pred[0])})

@app.route('/predict', methods=['POST'])
def create_task():
    if request.method == 'POST':
       
        t_name = request.json['news']
        sample=[t_name]
        sample=tfidf_vectorizer.transform(sample).toarray()

        print(sample)
        pred=clf.predict(sample)

        return jsonify({"pred":int(pred[0])})

app.run()