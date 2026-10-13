"""Explicit text queries and instance masks, independent of taxonomy labels."""
import json


class QueryJsonLoader:
    def __init__(self, annotation_file):
        with open(annotation_file, encoding='utf-8') as handle:
            payload = json.load(handle)
        if payload.get('format') != 'sam3_sft_queries_v1':
            raise ValueError('Expected sam3_sft_queries_v1')
        self.images = payload['images']
        if not self.images:
            raise ValueError('Dataset is empty')
        seen = set()
        for item in self.images:
            if item['id'] in seen:
                raise ValueError('Duplicate image ID')
            seen.add(item['id'])
            w, h = item['width'], item['height']
            if w <= 0 or h <= 0:
                raise ValueError('Invalid image dimensions')
            ids = set()
            for obj in item['objects']:
                if obj['id'] in ids:
                    raise ValueError('Duplicate object ID')
                ids.add(obj['id'])
                x, y, bw, bh = obj['bbox']
                if not (bw > 0 and bh > 0 and x >= 0 and y >= 0 and x + bw <= w and y + bh <= h):
                    raise ValueError('bbox must be pixel xywh within the image')
                mask = obj['segmentation']
                if not isinstance(mask, dict) or mask.get('size') != [h, w] or 'counts' not in mask:
                    raise ValueError('Expected COCO RLE with size [height, width]')
            if not item['queries']:
                raise ValueError('Each image requires queries')
            for query in item['queries']:
                if not query['text'].strip() or not set(query['object_ids']) <= ids:
                    raise ValueError('Empty query or missing object reference')
                if not query['object_ids'] and not query.get('is_exhaustive', False):
                    raise ValueError('Negative queries require confirmed absence: is_exhaustive=true')

    def getDatapointIds(self):
        return list(range(len(self.images)))

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        item = self.images[idx]
        w, h = item['width'], item['height']
        annotations = []
        for obj in item['objects']:
            x, y, bw, bh = obj['bbox']
            annotations.append(dict(id=obj['id'], object_id=obj['id'], image_id=0,
                bbox=[x/w, y/h, bw/w, bh/h], area=obj.get('area', bw*bh),
                segmentation=obj['segmentation'], is_crowd=bool(obj.get('iscrowd', False))))
        queries = []
        for qid, query in enumerate(item['queries']):
            queries.append(dict(id=qid, original_cat_id=qid, object_ids_output=query['object_ids'],
                query_text=query['text'], query_processing_order=0, ptr_x_query_id=None,
                ptr_y_query_id=None, image_id=0, input_box=None, input_box_label=None,
                input_points=None, is_exhaustive=query.get('is_exhaustive', True)))
        return queries, annotations

    def loadImagesFromDatapoint(self, idx):
        item = self.images[idx]
        return [dict(id=0, file_name=item['file_name'], original_img_id=item['id'], coco_img_id=item['id'])]
