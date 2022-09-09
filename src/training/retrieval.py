import logging

import os
import pdb
import pickle

import torch
import numpy
import time
import numpy as np
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from contextlib import suppress
from open_clip.electra_utils import tokenize


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def encode_data(model, dataloader, args):
    """Encode all images and captions loadable by `data_loader`
    """
    print("Evaluating...")
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    with torch.no_grad():

        for i, (images, captions, index, image_name) in tqdm(enumerate(dataloader)):
            batch_size = images.shape[0]
            captions = torch.cat([tokenize(c) for c in captions])

            images = images.to(args.device)
            captions = captions.to(args.device)
            with autocast():


                if args.distributed and not args.horovod:
                    img_emb = model.module.encode_image(images)
                    cap_emb = model.module.encode_text(captions)
                else:
                    img_emb = model.encode_image(images)
                    cap_emb = model.encode_text(captions)

                # import pdb; pdb.set_trace()
                # initialize the numpy arrays given the size of the embeddings
                if img_embs is None:
                    img_embs = np.zeros((len(dataloader.dataset), img_emb.size(1)))
                    cap_embs = np.zeros((len(dataloader.dataset), cap_emb.size(1)))

                # preserve the embeddings by copying from gpu and converting to numpy
                for idx in range(batch_size):
                    img_embs[i * batch_size + idx] = img_emb.data.cpu().numpy().copy()[idx]
                    cap_embs[i * batch_size + idx] = cap_emb.data.cpu().numpy().copy()[idx]

        del images, captions

    return img_embs, cap_embs



def run(model, dataloader, args):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """


    print('Computing results...')
    img_embs, cap_embs = encode_data(model, dataloader, args)


    # evaluation
    if args.data_name == 'wiki':
        npts = 1
        caps_per_image = 2
    else:
        npts = None
        caps_per_image = 5

    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / caps_per_image, cap_embs.shape[0]))

    pdb.set_trace()
    r, rt = i2t(img_embs, cap_embs, return_ranks=True, npts=npts)
    ri, rti = t2i(img_embs, cap_embs, return_ranks=True, npts=npts)
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.1f" % rsum)
    print("Average i2t Recall: %.1f" % ar)
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
    print("Average t2i Recall: %.1f" % ari)
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)

    return r, ri


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        caps_per_image = 5
    else:
        # Wiki
        caps_per_image = 2

    npts = images.shape[0] / caps_per_image

    index_list = []
    npts = int(npts)
    # import pdb; pdb.set_trace()
    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[caps_per_image * index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(caps_per_image * index, caps_per_image * index + caps_per_image, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        caps_per_image = 5
    else:
        # Wiki
        caps_per_image = 2

    npts = images.shape[0] / caps_per_image

    ims = numpy.array([images[i] for i in range(0, len(images), caps_per_image)])
    npts = int(npts)

    ranks = numpy.zeros(caps_per_image * npts)
    top1 = numpy.zeros(caps_per_image * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[caps_per_image * index:caps_per_image * index + caps_per_image]

        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[caps_per_image * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[caps_per_image * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)




def retrieval_eval(model, data, epoch, args):
    if 'mscoco-val' not in data and 'f30k-val' not in data:
        return {}
    if args.retrieval_frequency == 0:
        return {}
    if (epoch % args.retrieval_frequency) != 0 and epoch != args.epochs:
        return {}

    logging.info('Starting retrieval coco.')



    results = {}
    if 'mscoco-val' in data:
        ITresults, TIresults = run(model, data['mscoco-val'].dataloader, args)
        results['mscoco5k-ImageTextRetrieval-recall@1'] = ITresults[0]
        results['mscoco5k-ImageTextRetrieval-recall@5'] = ITresults[1]
        results['mscoco5k-ImageTextRetrieval-recall@10'] = ITresults[2]
        results['mscoco5k-TextImageRetrieval-recall@1'] = TIresults[0]
        results['mscoco5k-TextImageRetrieval-recall@5'] = TIresults[1]
        results['mscoco5k-TextImageRetrieval-recall@10'] = TIresults[2]



    logging.info('Finished retrieval coco.')

    return results
