import os
import sys
import argparse
import torch
import pandas as pd
import csv

sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_file", type=str, default="pretrained_models/atomic_pretrained_model.pickle" \
                        , help="path to model pre-trained")
    parser.add_argument("--sampling_algorithm", type=str, default="greedy")
    parser.add_argument("--dataset", type=str, default="atomic", help="choose atomic or conceptnet")
    parser.add_argument("--relation", type=str, default="all", help="relation types")
    parser.add_argument("--input_file", type=str, default="inputs/test.tsv", help="path to input file")
    parser.add_argument("--output_file", type=str, default="outputs/test_out.csv", help="path to save output")

    args = parser.parse_args()

    opt, state_dict = interactive.load_model_file(args.model_file)

    data_loader, text_encoder = interactive.load_data(args.dataset, opt)

    if args.dataset == 'atomic':
        n_ctx = data_loader.max_event + data_loader.max_effect
    else:
        n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
    n_vocab = len(text_encoder.encoder) + n_ctx
    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)
    
    #read file input
    inputs = pd.read_csv(args.input_file, sep='\t')
    sentences = inputs['text']

    if args.device != "cpu":
        cfg.device = int(args.device)
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        model.cuda(cfg.device)
    else:
        cfg.device = "cpu"

    if args.dataset == "atomic":
        relations = ['oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant']
        if not os.path.exists(args.output_file):
            with open(args.output_file, 'a', newline='') as out:
                writer = csv.writer(out)
                writer.writerow(['text', 'oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant'])
        sampler = interactive.set_sampler(opt, args.sampling_algorithm, data_loader)
        for s in sentences:
            output = interactive.get_atomic_sequence(
                s, model, sampler, data_loader, text_encoder, args.relation)
            row = [s]
            for r in relations:
                row.append(output[r]['beams'])
            print(row)
            with open(args.output_file, 'a', newline='') as out:
                writer = csv.writer(out)
                writer.writerow(row)
    
    else:
        relations = ['AtLocation', 'CapableOf', 'Causes', 'CausesDesire', 'CreatedBy' \
            , 'DefinedAs', 'DesireOf', 'Desires', 'HasA', 'HasFirstSubevent', 'HasLastSubevent' \
            , 'HasPainCharacter', 'HasPainIntensity', 'HasPrerequisite', 'HasProperty' \
            , 'HasSubevent', 'InheritsFrom', 'InstanceOf', 'IsA', 'LocatedNear', 'LocationOfAction' \
            , 'MadeOf', 'MotivatedByGoal', 'NotCapableOf', 'NotDesires', 'NotHasA' \
            , 'NotHasProperty', 'NotIsA', 'NotMadeOf', 'PartOf', 'ReceivesAction' \
            , 'RelatedTo', 'SymbolOf', 'UsedFor']
        sampler = interactive.set_sampler(opt, args.sampling_algorithm, data_loader)
        if not os.path.exists(args.output_file):
            with open(args.output_file, 'a', newline='') as out:
                writer = csv.writer(out)
                header = ['text']
                header.extend(relations)
                writer.writerow(header)
        for s in sentences:
            output = interactive.get_conceptnet_sequence(
                s, model, sampler, data_loader, text_encoder, args.relation)
            row = [s]
            for r in relations:
                row.append(output[r]['beams'])
            print(row)
            with open(args.output_file, 'a', newline='') as out:
                writer = csv.writer(out)
                writer.writerow(row)
    