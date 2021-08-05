import json
import logging
import os
import random

from rich import print
from rich.progress import track
from shutil import copyfile
from speechbrain.dataio.dataio import read_audio
from tqdm import tqdm

logger = logging.getLogger(__name__)

def prepare_data(
    Ydata_folder,
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    split_ratio=[80, 90, 100],
    different_speakers=False,
    seed=12,
):
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    SAMPLERATE = 16000
    # 读取所有的session
    sessions = os.listdir(Ydata_folder)
    sumdialoog = 0
    all_json = {}

    for se in tqdm(sessions):
        # 进来找到所有的对话(对话表示整个语音，句子表示对话内的一句一句话)
        dialog = os.listdir(os.path.join(Ydata_folder,se,'sentences','wav'))
        for digindex, dlgname in enumerate(dialog):
            sumdialoog +=1
            # 把diglog复制到文件夹里
            savedialog = os.path.join(data_folder,dlgname)
            dialogsen = os.path.join(savedialog,'sentences')
            dialogall = os.path.join(savedialog,'dialogall')
            if not os.path.exists(dialogsen):
                os.makedirs(dialogsen)
            if not os.path.exists(dialogall):
                os.makedirs(dialogall)
            #先把dialog复制进去
            if not os.path.exists(os.path.join(dialogall,dlgname+'.wav')):
                copyfile(os.path.join(Ydata_folder,se,'dialog','wav',dlgname+'.wav'),os.path.join(dialogall,dlgname+'.wav'))
            #再把文字内容读取一下
            with open(os.path.join(Ydata_folder,se,'dialog','transcriptions',dlgname+'.txt')) as ts:
                lg = ts.readlines()

            #再把标记内容读取一下
            ann = []
            with open(os.path.join(Ydata_folder,se,'dialog','EmoEvaluation',dlgname+'.txt')) as an:
                aln = an.readlines()
                # 进行一次过滤只保存标记的内容的第一行
                for l in aln:
                    if l.find('[')!=-1:
                        ann.append(l)
            json_dict = {}
            json_file = os.path.join(data_folder,dlgname,'ann_sentence.json')
            #再循环遍历sentence,进行文件复制，标签对齐
            for sentencesfile in os.listdir(os.path.join(Ydata_folder,se,'sentences','wav',dlgname)):
                if sentencesfile.split(".")[-1]=='wav':
                    sentencesname = sentencesfile.split('.')[0]
                    contxt = ''
                    eml = ''
                    # 复制文件到
                    if not os.path.exists(os.path.join(dialogsen,sentencesfile)):
                        copyfile(os.path.join(Ydata_folder,se,'sentences','wav',dlgname,sentencesfile),os.path.join(dialogsen,sentencesfile))
                    # 查找对应的语言内容
                    for nr in lg:
                        if nr.find(sentencesname)!=-1:
                            contxt = nr.split(':')[1].replace("\n",'')

                    # 查找对应的感情标签
                    for e in ann:
                        if e.find(sentencesname)!=-1:
                            eml = e.split('\t')[2]

                    # 读取音频时长
                    signal = read_audio(os.path.join(Ydata_folder,se,'sentences','wav',dlgname,sentencesfile))
                    duration = signal.shape[0] / SAMPLERATE
                    relative_path = os.path.join("{data_root}",dlgname,"sentences",sentencesfile)
                    json_dict[sentencesname] = {
                        "wav": relative_path,
                        "length": duration,
                        "emo": eml,
                        "content": contxt
                    }
                    all_json[sentencesname] = {
                        "wav": relative_path,
                        "length": duration,
                        "emo": eml,
                        "content": contxt
                    }

            with open(json_file, mode="w") as json_f:
                json.dump(json_dict, json_f, indent=2)

    random.seed(seed)
    # 对alljson进行随机打乱和划分三种数据集
    dict_key_ls = list(all_json.keys())
    random.shuffle(dict_key_ls)

    train_json = {}
    test_json = {}
    valid_json = {}

    for i in range(len(dict_key_ls)):
        if i< int(len(dict_key_ls)*split_ratio[0]*0.01):
            train_json[dict_key_ls[i]] =  all_json.get(dict_key_ls[i])
        elif i>=int(len(dict_key_ls)*split_ratio[0]*0.01) and i<int(len(dict_key_ls)*split_ratio[1]*0.01):
            test_json[dict_key_ls[i]] = all_json.get(dict_key_ls[i])
        else:
            valid_json[dict_key_ls[i]] = all_json.get(dict_key_ls[i])

    print("\nsum of dialog",str(sumdialoog))
    print("sum of sentence",str(len(all_json)))
    print("sum of train_sentence",str(len(train_json)))
    print("sum of test_sentence",str(len(test_json)))
    print("sum of vaild_sentence",str(len(valid_json)))

    with open(save_json_train, mode="w") as json_f:
        json.dump(train_json, json_f, indent=2)
    with open(save_json_valid, mode="w") as json_f:
        json.dump(valid_json, json_f, indent=2)
    with open(save_json_test, mode="w") as json_f:
        json.dump(test_json, json_f, indent=2)

def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True

def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True

if __name__ == '__main__':
    save_json_train = './train.json'
    save_json_valid = './valid.json'
    save_json_test = './test.json'
    data_folder = './data'
    Ydata_folder = r'F:\DATA\au\IEMOCAP'
    prepare_data(Ydata_folder,data_folder,save_json_train,save_json_valid,save_json_test)
