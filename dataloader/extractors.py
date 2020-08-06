import os
from collections import Counter
from random import shuffle
from typing import List

from tqdm.autonotebook import tqdm

from dataloader.data import MIMICDataset, TabularFeature, get_tables
from dataloader.utils import Event


def extract_sentences(dataset, tables: List[TabularFeature], event_class=Event, suffix='', **params):
    token_counter = Counter()
    with open(f'embeddings/sentences{suffix}.txt', 'w') as f:
        for sample in tqdm(dataset, desc='extract sentences'):
            for table in tables:
                for t, step in sample['inputs'][table.table]:
                    sentence = set()
                    for label, value in step:
                        event = event_class(table, label, value, time=t, **params)
                        sentence.add(event.text)
                    if sentence:
                        f.write(' '.join(sentence) + '\n')
                    token_counter.update(sentence)
            # seperate patients
            f.write('\n')

    os.makedirs('embeddings', exist_ok=True)
    with open(f'embeddings/sentences{suffix}.txt.counts', 'w') as f:
        for token, count in token_counter.most_common():
            f.write(f'{token} {count}\n')

    return token_counter


if __name__ == '__main__':
    DEVICE = 'cpu'
    dem, chart, lab, output = get_tables(['CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS', 'dem'],
                                         load=False,
                                         n_bins=12)

    from dataloader.labels import get_labels
    labels = get_labels(DEVICE)
    tables = [chart, lab, output, dem]
    # Use train and validation set to generate sentences for fasttext
    train_set = MIMICDataset(datalist_file='train_listfile.csv',
                             base_path='mimic3-benchmarks/data/multitask',
                             datasplit='train', mode='EVAL',
                             tables=tables, labels=labels)

    # generate cache
    for table in tables:
        table.fit(train_set)
        table.save()

    tables = [chart, lab, output]
    extract_sentences(train_set,
                      tables,
                      suffix='.mimic3',
                      event_class=Event)
