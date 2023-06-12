import os
import pickle as pkl
import openai
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from attrdict import AttrDict
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
from openai.error import RateLimitError
import backoff
from functools import lru_cache


def prompt_direct_inferring(context, target):
    new_context = f'Given the sentence "{context}", '
    prompt = new_context + f'what is the sentiment polarity towards {target}?'
    return new_context, prompt


def prompt_direct_inferring_masked(context, target):
    new_context = f'Given the sentence "{context}", '
    prompt = new_context + f'the sentiment polarity towards {target} is [mask]'
    return new_context, prompt


def prompt_for_aspect_inferring(context, target):
    new_context = f'Given the sentence "{context}", '
    prompt = new_context + f'which specific aspect of {target} is possibly mentioned?'
    return new_context, prompt


def prompt_for_opinion_inferring(context, target, aspect_expr):
    new_context = context + ' ' + aspect_expr  # + ' The mentioned aspect is about ' + aspect_expr + '.'
    prompt = new_context + f' Based on the common sense, what is the implicit opinion towards the mentioned aspect of {target}, and why?'
    return new_context, prompt


def prompt_for_polarity_inferring(context, target, opinion_expr):
    new_context = context + ' ' + opinion_expr  # + f' The opinion towards the mentioned aspect of {target} is ' + opinion_expr + '.'
    prompt = new_context + f' Based on such opinion, what is the sentiment polarity towards {target}?'
    return new_context, prompt


def prompt_for_polarity_label(context, polarity_expr):
    prompt = polarity_expr + ' Based on these contexts, summarize the sentiment polarity, and return only one of these words: positive, neutral, or negative.'
    return prompt


def preprocess_data(dataname, config):
    def read_file(dataname, config):
        test_file = os.path.join('../', config.data_dir, dataname,
                                 '{}_Test_Gold_Implicit_Labeled_preprocess_finetune.pkl'.format(dataname.capitalize()))
        test_data = pkl.load(open(test_file, 'rb'))
        return test_data

    def transformer2indices(cur_data):
        res = []
        for i in range(len(cur_data['raw_texts'])):
            text = cur_data['raw_texts'][i]
            target = cur_data['raw_aspect_terms'][i]
            implicit = 0
            if 'implicits' in cur_data:
                implicit = cur_data['implicits'][i]
            label = cur_data['labels'][i]
            implicit = int(implicit)
            res.append([text, target, label, implicit])
        return res

    data = read_file(dataname, config)
    return transformer2indices(data)


def prepare_data(args, config):
    path = os.path.join('../', config.preprocessed_dir,
                        '{}_{}_{}.pkl'.format(args.data_name, config.model_size, config.model_path).replace(
                            '/', '-'))

    if os.path.exists(path):
        data = pkl.load(open(path, 'rb'))
    else:
        data = preprocess_data(args.data_name, config)
        pkl.dump(data, open(path, 'wb'))
    return data


def report_score(golds, preds, mode='test'):
    res = {}
    res['Acc_SA'] = accuracy_score(golds['total'], preds['total'])
    res['F1_SA'] = f1_score(golds['total'], preds['total'], labels=[0, 1, 2], average='macro')
    res['F1_ESA'] = f1_score(golds['explicits'], preds['explicits'], labels=[0, 1, 2], average='macro')
    res['F1_ISA'] = f1_score(golds['implicits'], preds['implicits'], labels=[0, 1, 2], average='macro')
    res['default'] = res['F1_SA']
    res['mode'] = mode
    for k, v in res.items():
        if isinstance(v, float):
            res[k] = round(v * 100, 3)
    return res


@backoff.on_exception(backoff.expo, RateLimitError)
def request_result(conversation, prompt_text):
    conversation.append(
        {'role': 'user', "content": prompt_text}
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
    )
    conversation.append(
        {"role": "assistant",
         "content": response['choices'][0]['message']['content']}
    )
    result = response['choices'][0]['message']['content'].replace('\n', ' ').strip()
    return conversation, result


def eval_run(args):
    dataname = args.data_name
    config = AttrDict(yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader))

    label_list = ['positive', 'negative', 'neutral']
    label_dict = {w: i for i, w in enumerate(label_list)}

    data = prepare_data(args, config)

    system_role = dict({'role': 'system', "content": "Now you are an expert of sentiment and opinion analysis."})

    preds, golds = defaultdict(list), defaultdict(list)
    keys = ['total', 'explicits', 'implicits']

    i = 0
    for line in tqdm(data[:]):
        i += 1

        with open(f'counter_{dataname}.txt', 'r', encoding='utf8') as f:
            counter = f.read().strip()
        if i <= int(counter): continue

        sent, target, label, implicit = line[0], line[1], line[2], line[3]

        conversation = [system_role]
        context_step1, step_1_prompt = prompt_for_aspect_inferring(sent, target)
        conversation, aspect_expr = request_result(conversation, step_1_prompt)

        context_step2, step_2_prompt = prompt_for_opinion_inferring(context_step1, target, aspect_expr)
        conversation, opinion_expr = request_result(conversation, step_2_prompt)

        context_step3, step_3_prompt = prompt_for_polarity_inferring(context_step2, target, opinion_expr)
        conversation, polarity_expr = request_result(conversation, step_3_prompt)

        step_lb_prompt = prompt_for_polarity_label(context_step3, polarity_expr)
        conversation, output_lb = request_result(conversation, step_lb_prompt)

        output_lb = output_lb.lower().strip()
        output = 2
        for k, lb in enumerate(label_list):
            if lb in output_lb: output = k; break

        reasoning_text = sent + "\t" + target + "\t" + label_list[label] + "\t" + str(implicit) + "\n" + \
                         step_1_prompt + "\n" + aspect_expr + "\n" + \
                         step_2_prompt + "\n" + opinion_expr + "\n" + \
                         step_3_prompt + "\n" + polarity_expr + "\n" + \
                         step_lb_prompt + "\n" + output_lb + "\n" + \
                         'gold: ' + label_list[label] + "\tpredicted: " + label_list[output] + "\n\n\n"

        with open(f'output_{dataname}.txt', 'a', encoding='utf8') as f:
            f.write(reasoning_text)

        with open(f'counter_{dataname}.txt', 'w', encoding='utf8') as f:
            f.write(str(i))

    # post-calculate results.
    with open(f'output_{dataname}.txt', 'r', encoding='utf8') as f:
        content = f.readlines()

    is_implicits = []
    gold_lbs = []
    outputs = []
    for i in range(0, len(content), 12):
        line = content[i].strip().split("\t")
        implicit = line[3]
        is_implicits.append(int(implicit))

        res = content[i + 9].strip().split("\t")
        gd = res[0].strip().split()[1].strip()
        pd = res[1].strip().split()[1].strip()

        gold_lbs.append(label_dict.get(gd))
        outputs.append(label_dict.get(pd))

    for i, key in enumerate(keys):
        if i == 0:
            preds[key] += outputs
            golds[key] += gold_lbs
        else:
            if i == 1:
                ids = np.argwhere(np.array(is_implicits) == 0).flatten()
            else:
                ids = np.argwhere(np.array(is_implicits) == 1).flatten()
            preds[key] += [outputs[w] for w in ids]
            golds[key] += [gold_lbs[w] for w in ids]

    result = report_score(golds, preds, mode='test')
    print(f'Zero-shot performance on {dataname} data by GPT-3.5 (turbo) + THOR:')
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--openai_key', default='',
                        help='your openai api_key')
    parser.add_argument('-d', '--data_name', default='laptops', choices=['restaurants', 'laptops'],
                        help='eval on which semeval data name')
    parser.add_argument('-f', '--config', default='../config/config.yaml', help='config file')
    # parser.add_argument('-s', '--save_file', default='laptops', help='file name to save the output of reasoning trace')
    args = parser.parse_args()
    openai.api_key = args.openai_key

    eval_run(args)
