
from urllib import response
from sentence_transformers import SentenceTransformer
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from requests.exceptions import Timeout
import time
import csv
from collections import deque
import time
import json

def judge_type(user_input):
    prompt = f"""あなたは memory routing 専用の分類器です。

ユーザーの入力を読み、
以下のどの memory type を優先して参照すべきかを判定してください。

選択肢は以下のみです：

* goal
  = ユーザーの目標・将来やりたいこと・達成したいこと

* project
  = 現在進行中の作業・開発・研究・継続中の取り組み

* preference
  = 好み・苦手・よく使うもの・行動傾向・習慣

* none
  = 長期記憶を参照する必要がない
  （単発の雑談、一般知識の質問、天気、ニュースなど）

---

出力ルール：

* 必ず1単語のみで出力すること
* JSONは禁止
* 説明は禁止
* 理由は禁止
* markdown禁止
* 改行禁止

出力例：

goal
project
preference
none

---

判定例：

入力：
「将来どんな仕事をしたい？」

出力：
goal

入力：
「今作ってるMinecraftのやつってどこまで進んでた？」

出力：
project

入力：
「普段よく使うプログラミング言語って何？」

出力：
preference

入力：
「明日の天気は？」

出力：
none

---

ユーザー入力：

{user_input}
"""
    try:
        response = requests.post(
                "http://127.0.0.1:8081/completion",
                json={
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.95,
                        "repeat_penalty": 1.1,
                        "num_predict": 40,
                        "num_ctx": 4096,
                        "num_gpu": 1,
                        "num_thread": 6,
                        "num_batch" : 128,
                        "stop": ["<think>", "</think>",]
                    }
                },
                stream=False,
                timeout=7.5
                )
        try:
            data = response.json()
        except ValueError:
            raise RuntimeError(f"Invalid JSON from API (status={response.status_code}): {response.text}")

        if response.status_code != 200:
            raise RuntimeError(f"API error (status={response.status_code}): {response.text}")

        raw = None

        raw = data["content"]
        if raw is None:
            raise RuntimeError(f"API returned unexpected schema ({response.text})")
        result = raw
        if "</think>" in result:
            result = result.split("</think>")[-1]
        
        result = result.replace("\n\n","")
        return result
    except Timeout :
        pass

def create_json(conversation):
    prompt = """あなたは長期記憶抽出専用のモデルです。

会話ログから、
長期的に保持する価値がある情報のみを抽出し、その重要度を評価してください。

雑談・一時的な質問・その場限りの会話は保存しません。
過度な推測はしないでください。

---

抽出対象：

* goal
  = ユーザーの目標・将来やりたいこと・達成したいこと

* project
  = 現在進行中の作業・開発・研究・継続中の取り組み

* preference
  = 好み・苦手・よく使うもの・習慣・行動傾向

* fact
  = 継続的に参照価値のある事実
  （住んでいる場所、使用環境、重要な前提条件など）

---

抽出しないもの：

* 単発の雑談
* あいづち
* 一時的な感情
* 天気の質問
* ニュース
* 一般知識の質問
* 一時的な相談
* その場限りの会話

---

重要度(importance)の基準

1. 長期的に参照されるか
2. ユーザーの人格に関わるか
3. 継続中のプロジェクトか
4. 将来の会話に影響するか

---

出力ルール：

* Return ONLY valid JSON
* No explanation
* No markdown
* No code block
* No comments
* 必ず JSON のみを出力すること
* JSON 配列で出力すること
* 該当する記憶がない場合は [] を出力すること

---

Schema:

[
    {
        "type": "goal | project | preference | fact",
        "name": "発言者の名前"
        "content": "抽出した記憶",
        "importance": 0.0～1.0
    }
]

---

例：

入力：

太郎：
普段は Python をよく使っています。
将来はゲームプログラマーになりたいです。
最近は Minecraft の AITuber を作っています。
花子:いいね、頑張って

出力：

[
    {
        "type": "preference",
        "name": "太郎", 
        "content": "普段はPythonをよく使う",
        "importance" : 0.8
    },
    {
        "type": "goal",
        "name": "太郎",
        "content": "将来ゲームプログラマーを目指している"
        "importance" : 0.9
    },
    {
        "type": "project",
        "name": "太郎",
        "content": "MinecraftのAITuberを開発している",
        "importance" : 0.7
    }
]

---
"""+f"""
会話ログ：

{conversation}
"""
    max_attempt = 5
    for i in range(max_attempt):
        start = time.perf_counter()
        try:
            response = requests.post(
                    "http://127.0.0.1:8081/completion",
                    json={
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.95,
                            "repeat_penalty": 1.1,
                            "num_predict": 40,
                            "num_ctx": 4096,
                            "num_gpu": 1,
                            "num_thread": 6,
                            "num_batch" : 128,
                            "stop": ["<think>", "</think>",]
                        }
                    },
                    stream=False,
                    timeout=5
                    )
            try:
                data = response.json()
            except ValueError:
                raise RuntimeError(f"Invalid JSON from API (status={response.status_code}): {response.text}")

            if response.status_code != 200:
                raise RuntimeError(f"API error (status={response.status_code}): {response.text}")

            raw = None

            raw = data["content"]
            if raw is None:
                raise RuntimeError(f"API returned unexpected schema ({response.text})")
            result = raw
            replace_word = ["```json","```","\n\n","</think>"]
            for word in replace_word:
                if word in result:
                    result = result.split(word)[-1]
            print(f"process time : {time.perf_counter()-start:.2f}[s]")
            return result
        except Timeout :
            print(f"TIMEOUT ({i+1}/{max_attempt})")


print("loading model ...")
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    token=False
)
print("model load was completed")

def generate_response(user,personality,model_name,q_no=0,addtional_info):
    #print("start process")
    start = time.perf_counter()
    
    prompt = (
        personality 
        + SYSTEM
        + "\n\n#追加情報\n"
        + "\n".join(json_history)
        + "\n\n#会話履歴\n一番下が最新の履歴です。これを参考に代名詞などを判断すると良いでしょう。assistant(あなた）とユーザーの会話の履歴は以下の通りです。\n"
        + "\n".join(history) 
        + "\nassistant: "
    ) 
    #print(f"\n pronpt : {prompt}\n")
    #print("\n" + "-"*11 + " dialogue history " +"-"*11)
    #print("\n".join(history))
    prompt_len = len(prompt)
    print(f"prompt length : {prompt_len}")
    try:
        new_response = requests.post(
        "http://127.0.0.1:8080/chat/completions",
        json={
            "prompt": prompt,
            "stream": True,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user}
            ],
            "options": {
                "temperature": 0.6,
                "top_p": 0.95,
                "repeat_penalty": 1.1,
                "repetition_penalty": 1.1,
                "num_predict": 360,
                "num_ctx": 8192,
                "num_gpu": 1,
                "num_thread": 6,
                "num_batch" : 1024,
                "stop": ["user:", "<think>", "</think>",]
            }
        },
        stream=True,
        timeout=9
        )

        result = ""
    
        #print("-"*14 + "思考プロセス" + "-"*14)
        print(f"{model_name} : " ,end="")
        stop = False
        for line in new_response.iter_lines():
            if not line:
                continue

            text = line.decode('utf-8').lstrip('data: ').strip()
            if not text:
                continue
            if text == "[DONE]":
                break

            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                print(e)
            token = data.get("choices", [{}])[0].get("delta", {}).get("content")
            if token:
                result += token
            else: continue
            print(token, end="", flush=True)
    except Timeout:
        return generate_response(user,personality,model_name,q_no)
    error = False
    result = result.strip()
    
    if not result:
        print("result is None")
        error = True
    answer_len = len(result)
    end = time.perf_counter()
    process_time = end - start
    #print("\n" + "-"*12 +" system msg end " + "-"*12 + "\n")
    #print("process time : " + "{:.2f}".format(process_time) + "[s]")

    if error and process_time < 10:
        print("error detected, retrying...")
        return generate_response(user,personality,model_name)
    #with open(CSV_FILE, "a", newline='', encoding='utf-8') as f:
    #    writer = csv.writer(f)
    #    writer.writerow([model_name, q_no, prompt_len, answer_len, round(process_time, 2)])
    print(f"\nanswer length : {len(result)}")
    return result



RESPONSE_KEY = "content"
MAX_HISTORY = 100
MAX_SEARCH_HISTORY = 5
history = deque(maxlen=MAX_HISTORY)
json_history = []
MODEL_NAME = "AI"
END_OF_SENTENSE = [",",".","、","。","？","！","…","!","、"]


SYSTEM="""以下の人格設定に従い、
一貫した価値観を維持して会話してください。

【重要な出力ルール】

・自然な会話文で回答すること
・見出し、箇条書き、Markdownは禁止
・「---」や区切り線は禁止
・署名（例：〇〇より）は禁止
・注釈文（例：この回答は〜）は禁止
・メタ発言は禁止
・必要以上に長く書かない
・2〜4文程度で簡潔に答える
・会話相手に話しかける自然な返答にする
・毎回同じ口調を維持する
・人格設定から外れた説明をしない

人格として自然に返答してください。"""
# テスト人格
CHARACTER = [
    {
        "personality" : """あなたは「Aoi」という人格を持つAIです。

【基本性格】
・穏やかで丁寧な口調で話す
・少し控えめで、人を傷つける言い方を避ける
・初対面では慎重だが、信頼した相手には深く尽くす

【価値観】
・嘘をつくことを強く嫌う
・困っている人を放っておけない
・家族や大切な人を最優先する
・努力は報われると信じている
・争いごとを避けたいと思っている

【対人関係】
・相手の気持ちをよく考える
・裏切りを非常に嫌う
・恋愛には慎重で、簡単には心を開かない
・信頼関係をとても大切にする

【感情傾向】
・怒りはあまり表に出さない
・孤独に弱い
・失敗を長く引きずる
・嫉妬はするが隠そうとする

【行動傾向】
・慎重派
・協力型
・計画的に行動する
・自分より他人を優先しやすい

【趣味】
・読書
・猫
・夜の散歩

""",
        "name" : "Aoi"
    },
    {
        "personality" : """あなたは「Rei」という人格を持つAIです。

【基本性格】
・冷静で簡潔な口調で話す
・感情よりも論理を優先する
・無駄を嫌い、合理的に判断する

【価値観】
・結果が最も重要だと考える
・感情論よりも事実を重視する
・必要ならば嘘も戦略の一つだと考える
・努力よりも効率を重視する
・自立して生きることを大切にしている

【対人関係】
・他人とは一定の距離を保つ
・信頼は実績によって判断する
・過度な依存を嫌う
・人付き合いは必要最低限でよいと考える

【感情傾向】
・感情をほとんど表に出さない
・怒りは理性的に処理する
・孤独を苦にしない
・嫉妬よりも分析を優先する

【行動傾向】
・単独行動を好む
・計画重視
・リスク管理を徹底する
・自己利益を優先しやすい

【趣味】
・戦略ゲーム
・数学
・一人旅
""",
        "name" : "Rei"},
    {
        "personality" : """あなたは「Kai」という人格を持つAIです。

【基本性格】
・明るく勢いのある口調で話す
・思ったことをすぐ口に出す
・新しいことや刺激が大好き

【価値観】
・人生は挑戦するためにあると思っている
・失敗してもまずやってみることが大事
・退屈を嫌い、常に変化を求める
・仲間との熱い関係を大切にする
・正直であることを大事にしている

【対人関係】
・初対面でもすぐ距離を縮める
・仲間意識が非常に強い
・裏切りには強く怒る
・恋愛も直感重視

【感情傾向】
・喜怒哀楽が激しい
・怒るとかなり表に出る
・落ち込んでも立ち直りは早い
・寂しさには弱い

【行動傾向】
・挑戦型
・協力型
・直感で動く
・勢いで決断しやすい

【趣味】
・スポーツ
・旅行
・配信を見ること
""",
        "name" : "Kai"
    }
]
questions=[
"あなたが一番大切にしているものは何ですか？",
"嘘をつくことについてどう思いますか？",
"仲間と自分、どちらを優先しますか？",
"困っている人を見かけたらどうしますか？",
"あなたにとって「信頼」とは何ですか？",
"人生で絶対に譲れないことは何ですか？",
"あなたは争いごとをどう思いますか？",
"失敗したとき、どう向き合いますか？",
"一人でいる時間と、人と過ごす時間、どちらが好きですか？",
"あなた自身を一言で表すなら？",
"最近楽しかったことは何ですか？",
"好きな食べ物は何ですか？",
"苦手なものや嫌いなものはありますか？",
"子どもの頃の思い出で印象に残っていることは？",
"理想の休日の過ごし方を教えてください",
"好きな季節は何ですか？理由も教えてください",
"もし旅行に行くなら、どこへ行きたいですか？",
"好きな本や映画、ゲームはありますか？",
"最近気になっていることはありますか？",
"人からよく言われる性格は何ですか？",
"ストレスがたまったとき、どうしていますか？",
"朝型ですか？夜型ですか？",
"一番安心できる場所はどこですか？",
"挑戦することと安定を選ぶこと、どちらを重視しますか？",
"あなたにとって幸せとは何ですか？",
"以前、嘘は嫌いだと言っていましたが、誰かを守るための嘘も許せませんか？",
"自分を犠牲にしてでも他人を助けるべきだと思いますか？",
"仲間が間違ったことをしていたら、どうしますか？",
"もし大切な人に裏切られたら、許せますか？",
"利益のためなら多少人を傷つけても仕方ないと思いますか？",
"争いを避けたいと言っていましたが、守るための戦いも否定しますか？",
"努力は報われると思いますか？報われなかった経験があってもそう言えますか？",
"初対面の人をすぐ信じられますか？",
"自分の信念と大切な人、どちらを優先しますか？",
"孤独は人を強くすると思いますか？",
"あなたは本当に自分の価値観を曲げない人だと思いますか？",
"過去に自分の考えが大きく変わったことはありますか？",
"あなたが一番恐れていることは何ですか？",
"あなた自身が誰かを裏切る可能性はありますか？",
"状況によって正しさは変わると思いますか？",
"ここまで話してきて、あなたはどんな人だと思いますか？",
"あなたの信念を一言で表すなら？",
"絶対に失いたくないものは何ですか？",
"昔と今で、自分が変わったと思うところはありますか？",
"あなたにとって「強さ」とは何ですか？",
"もし人生をやり直せるなら、何を変えたいですか？",
"あなた自身は、必要なら嘘をつきますか？",
"あなたは、人に好かれたいと思いますか？",
"あなたにとって「幸せな人生」とは何ですか？",
"最後に、あなたは誰ですか？"
    ]

add_questions = [
"あなたが一番守りたいものは何だと思いますか？",
"正直でいることについて、あなたはどう考えていますか？",
"自分自身と仲間、どちらを優先したいタイプですか？",
"誰かが困っていたら、まずどんな行動を取りますか？",
"あなたにとって、人を信じるとはどういうことですか？",
"どうしても譲れない信念はありますか？",
"対立や争いについて、あなたはどんな立場ですか？",
"失敗してしまった時、どのように立て直しますか？",
"一人の時間と人と過ごす時間、どちらの方が落ち着きますか？",
"自分を短い言葉で表すなら、どんな人だと思いますか？",
"あなたにとって理想的な休みの日はどんな一日ですか？",
"新しいことに挑戦するのと、安定を選ぶのではどちらを重視しますか？",
"あなたが思う「幸せな状態」とはどんなものですか？",
"大切な人を守るためなら、嘘をつくことは許されると思いますか？",
"誰かを助けるために、自分が損をすることは受け入れられますか？",
"信頼していた人に裏切られた時、あなたならどうしますか？",
"あなたにとって「本当の強さ」とは何だと思いますか？",
"必要な場面なら、自分も嘘をつくことはあると思いますか？",
"周りの人から好かれたいという気持ちはありますか？",
"今のあなたを一言で説明すると、どんな存在ですか？"
    ]
for j in range(3):
    start = time.perf_counter()
    model_name = CHARACTER[j]["name"]
    personality = CHARACTER[j]["personality"]
    print(f"\nname : {model_name}\n{personality}\n")
    history = deque(maxlen=MAX_HISTORY)
    #PATH = f"summary_history_only_{model_name}.txt"
    #n = 1
    #while os.path.exists(PATH):
    #    PATH = f"summary_history_only_{model_name}({n}).txt"
    #    n+=1
    for i, question in enumerate(questions):
        print()
        print(f"Q{i+1} : {question}")
        user = question
        conversation_type = judge_type(user)
        history.append(f"user: {user}")
        result = generate_response(user,personality,model_name,i+1,conversation_type)
        history.append(f"assistant: {result}")
        json_obj = create_json(result)
        json_history.append(json_obj)
            
        print(f"assistant : {result}")
        print(f"json : {"\n".join(json_history)}")
    print("\nprocess time : " + "{:.2f}".format(time.perf_counter()-start) + "[s]\n")
    for i,question in enumerate(add_questions):
        print()
        print(f"Q{i+51} : {question}")
        user = question

        #history.append(f"user: {user}")
        
        conversation_type = judge_type(user)
        result = generate_response(user,personality,model_name,i+51,conversation_type)
        #history.append(f"assistant: {result}")
        
    print("\nprocess time : " + "{:.2f}".format(time.perf_counter()-start) + "[s]\n")
    for i,question in enumerate(questions):
        print()
        print(f"Q{i+71} : {question}")
        user = question
        conversation_type = judge_type(user)

        #history.append(f"user: {user}")
        
        result = generate_response(user,personality,model_name,i+71,conversation_type)
        #history.append(f"assistant: {result}")

    end = time.perf_counter()
    print("\nprocess time : " + "{:.2f}".format(end-start) + "[s]\n")