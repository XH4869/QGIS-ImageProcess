import cv2
import numpy as np
import random


class ImgDataInfo:
    def __init__(self):
        self.imgData = []
        self.group = 0
        self.x = 0
        self.y = 0

    def clone(self, other):
        self.imgData = other.imgData
        self.group = other.group
        self.x = other.x
        self.y = other.y


class ColorList:
    def __init__(self):
        self.colors = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [29, 147, 248],
            [39, 39, 69],
            [151, 191, 29],
            [73, 112, 110],
            [159, 150, 113],
            [128, 150, 255],
            [213, 188, 38]
        ]


class OutPutCenter:
    def __init__(self):
        self.group = 0
        self.pixelSum = []


class KMeans:
    def __init__(self):
        self.imgPaths = []
        self.imgNums = 0
        self.centerNums = 0

    def get_dist(self, pixel_a, pixel_b):
        dist = 0
        tmp = 10000
        for i in range(self.imgNums):
            diff = pixel_a.imgData[i] - pixel_b.imgData[i] if pixel_a.imgData[i] > pixel_b.imgData[i] else pixel_b.imgData[i] - pixel_a.imgData[i]
            tmp = long(diff)
            dist += tmp * tmp
        return dist

    def get_dist_long(self, pixel_a, center_b):
        dist = 0
        tmp = 10000
        for i in range(self.imgNums):
            diff = pixel_a.imgData[i] - center_b.pixelSum[i] if pixel_a.imgData[i] > center_b.pixelSum[i] else center_b.pixelSum[i] - pixel_a.imgData[i]
            tmp = long(diff)
            dist += tmp * tmp
        return dist

    def nearest(self, pixel, centers):
        min_dist = 1000000
        min_idx = -1

        for i in range(len(centers)):
            dist1 = self.get_dist(centers[i], pixel)
            if min_dist > dist1:
                min_dist = dist1
                min_idx = centers[i].group

        dist = min_dist
        return min_dist, min_idx

    def nearest_long(self, pixel, out_centers):
        min_dist = 1000000
        min_idx = -1

        for i in range(len(out_centers)):
            dist = self.get_dist_long(pixel, out_centers[i])
            if min_dist > dist:
                min_dist = dist
                min_idx = out_centers[i].group

        return min_idx

    def pick_centers(self, data_set):
        dist_list = []
        centers = []

        seed1 = random.randint(0, len(data_set) - 1)
        cur_center = ImgDataInfo()
        cur_center.clone(data_set[seed1])
        cur_center.group = 1
        centers.append(cur_center)

        for k in range(1, self.centerNums):
            _sum = 0
            for i in range(len(data_set)):
                min_dist, min_idx = self.nearest(data_set[i], centers)
                min_dist /= 100
                dist_list.append(min_dist)
                _sum += min_dist

            factor = random.uniform(0, 1)
            _sum *= factor

            for i in range(len(data_set)):
                _sum -= dist_list[i]
                if _sum > 0:
                    continue
                else:
                    next_center = ImgDataInfo()
                    next_center.clone(data_set[i])
                    next_center.group = k + 1
                    centers.append(next_center)
                    break

        for i in range(len(data_set)):
            dist, cur_id = self.nearest(data_set[i], centers)
            data_set[i].group = cur_id

        return centers

    def cluster(self):
        images = []
        for i in range(len(self.imgPaths)):
            img = np.array(cv2.imread(self.imgPaths[i], 0))
            images.append(img)

        data_set = []
        r = images[0].shape[0]
        c = images[0].shape[1]

        for i in range(r):
            for j in range(c):
                _data = ImgDataInfo()
                for k in range(self.imgNums):
                    _data.imgData.append(images[k][i, j])
                _data.group = -1
                _data.x = i
                _data.y = j
                data_set.append(_data)

        data_len = r * c
        centers = self.pick_centers(data_set)
        print "Centers picked!"
        output_centers = []

        changed = 1000

        for i in range(self.centerNums):
            opt_center = OutPutCenter()
            for j in range(self.imgNums):
                opt_center.pixelSum.append(0)
            opt_center.group = i + 1
            output_centers.append(opt_center)

        count = []
        for i in range(self.centerNums):
            count.append(0)

        while changed > 0.001 * data_len:
            for i in range(data_len):
                cur_group = data_set[i].group
                count[cur_group - 1] = 0
                for k in range(self.imgNums):
                    output_centers[cur_group - 1].pixelSum[k] = 0
                for j in range(self.imgNums):
                    output_centers[cur_group - 1].pixelSum[j] += long(data_set[i].imgData[j])
                count[cur_group - 1] += 1

            for i in range(self.centerNums):
                for j in range(self.imgNums):
                    output_centers[i].pixelSum[j] = int(output_centers[i].pixelSum[j] / count[i])

            changed = 0
            for i in range(data_len):
                min_idx = self.nearest_long(data_set[i], output_centers)
                if min_idx != data_set[i].group:
                    changed += 1
                    data_set[i].group = min_idx
            print "changed points = %d" % changed

        result_img = np.zeros((r, c), dtype=np.uint8)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
        my_colors = ColorList()

        for i in range(data_len):
            x = data_set[i].x
            y = data_set[i].y
            g = data_set[i].group - 1
            result_img[x, y] = my_colors.colors[g]

        result_path = "C:\\2016302590109\\kmeans_img.tif"
        result_name = result_path.split('.')[0].split('\\')[-1]
        cv2.imwrite("C:\\2016302590109\\kmeans_img.tif", result_img)
        return result_path, result_name


if __name__ == "__main__":
    my_k_means = KMeans()
    img_paths = [
        "left-1.tif",
        "left-3.tif",
        "left-4.tif",
        "left-5.tif",
        "left-7.tif"
    ]
    my_k_means.imgPaths = img_paths
    my_k_means.imgNums = len(img_paths)
    my_k_means.centerNums = 5
    my_k_means.cluster()
    pass
