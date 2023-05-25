import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os
import sys


def fourier_2d(gray_mat, n_val, m_val, inverse_bool):
    wn_val = 0
    wm_val = 0
    if inverse_bool:
        wn_val = math.exp(2 * math.pi / n_val) ** 1j
        wm_val = math.exp(2 * math.pi / m_val) ** 1j
    else:
        wn_val = math.exp(-2 * math.pi / n_val) ** 1j
        wm_val = math.exp(-2 * math.pi / m_val) ** 1j
    arr_left = []
    arr_right = []
    for n_idx in range(0, n_val):
        tmp_list = [0 for i in range(n_val)]
        for n_tmp in range(0, n_val):
            tmp_list[n_tmp] = wn_val ** (n_idx * n_tmp)
        arr_left.append(tmp_list)
    for m_idx in range(0, m_val):
        tmp_list = [0 for i in range(m_val)]
        for m_tmp in range(0, m_val):
            tmp_list[m_tmp] = wm_val ** (m_idx * m_tmp)
        arr_right.append(tmp_list)
    arr_left = np.array(arr_left)
    arr_right = np.array(arr_right)
    result = np.matmul(np.array(arr_left), gray_mat)
    result = np.matmul(result, np.array(arr_right))
    return result


def azimuthal_averaging(fouriered, n_val, m_val):
    half_n = n_val / 2
    half_m = m_val / 2
    distance_list = []
    distance_mat = [[0] * m_val for _ in range(n_val)]
    for n_entry in range(n_val):
        for m_entry in range(m_val):
            dist = math.sqrt((half_n - n_entry) ** 2 + (half_m - m_entry) ** 2)
            distance_mat[n_entry][m_entry] = dist
            distance_list.append(dist)
    distance_list = list(set(distance_list))
    distance_list.sort()
    distance_list = list(map(int, distance_list))
    max_dist = distance_list[-1]
    freq_sum_list = [0] * (max_dist + 1)
    freq_times_list = [0] * (max_dist + 1)
    for n_entry in range(n_val):
        for m_entry in range(m_val):
            freq_sum_list[int(distance_mat[n_entry][m_entry])] = freq_sum_list[int(distance_mat[n_entry][m_entry])] + \
                                                                 fouriered[n_entry][m_entry]
            freq_times_list[int(distance_mat[n_entry][m_entry])] = freq_times_list[
                                                                       int(distance_mat[n_entry][m_entry])] + 1
    for lentry in range(max_dist + 1):
        freq_sum_list[lentry] = freq_sum_list[lentry] / freq_times_list[lentry]

    return freq_sum_list


def highpass_filter(fouriered, n_val, m_val):
    half_n = n_val / 2
    half_m = m_val / 2
    for n_entry in range(n_val):
        for m_entry in range(m_val):
            dist = math.sqrt((half_n - n_entry) ** 2 + (half_m - m_entry) ** 2)
            if dist <= 45:
                fouriered[n_entry][m_entry] = 0
    return fouriered


def min_max_normalize(result):
    min = result[0][0]
    max = result[0][0]
    for entry in result:
        for real_entry in entry:
            if real_entry < min:
                min = real_entry
            if real_entry > max:
                max = real_entry
    result = ((result - min) * 1 / (max - min))
    return result


data_dir = sys.argv[1]
file_list = os.listdir(data_dir)
color_gray_list = []
fouriered_list = []
task2_list = []
for file in file_list:
    color_gray_list.append(cv2.imread(data_dir + '/' + file, cv2.IMREAD_GRAYSCALE))
# This is for Task1-2D Fourier Transform
print("Task1 start")
if not os.path.exists('./task1_result_dir'):
    os.mkdir('./task1_result_dir')
for idx, gray_entry in enumerate(color_gray_list):
    print(idx)
    n_val = gray_entry.shape[0]
    m_val = gray_entry.shape[1]
    tmp_result = fourier_2d(gray_entry, n_val, m_val, False)
    fouriered_list.append(tmp_result)
    tmp_result = np.log(np.abs(tmp_result))
    tmp_result = min_max_normalize(tmp_result)
    tmp_result = np.fft.fftshift(tmp_result)
    task2_list.append(tmp_result)
    cv2.imwrite('./task1_result_dir' + '/' + file_list[idx][0:-4] + '_task1.jpg', tmp_result * 255)
print("Task1 end")
# This is for Task2-Azimuthal averaging
print("Task2 start")
if not os.path.exists('./task2_result_dir'):
    os.mkdir('./task2_result_dir')
print("result by Azimuthal averaging")
for idx, task2_entry in enumerate(task2_list):
    n_val = task2_entry.shape[0]
    m_val = task2_entry.shape[1]
    task2_result = azimuthal_averaging(task2_entry, n_val, m_val)
    task2_result = task2_result[1:]
    task2_result = task2_result / task2_result[0]
    tmp = 0
    for task2_idx, entry in enumerate(task2_result):
        if task2_idx >= 150 and task2_idx < 200:
            tmp = tmp + 1 / entry
    if tmp / (50) < 2.55:
        result = True
    else:
        result = False
    print(str(idx) + ': ' + str(result))
    plt.plot(task2_result, label=file_list[idx])
plt.legend()
plt.xlim(-20, 250)
plt.ylim(0.25, 1.2)
plt.savefig('./task2_result_dir' + '/task2.png')
print("Task2 end")
# This is for Task3-High-pass filtering and 2D inverse Fourier Transform
print("Task3 start")
if not os.path.exists('./task3_result_dir'):
    os.mkdir('./task3_result_dir')
print("result by High-pass filtering and 2D inverse Fourier Transform")
for idx, fourier_entry in enumerate(fouriered_list):
    n_val = fourier_entry.shape[0]
    m_val = fourier_entry.shape[1]
    tmp_result = fourier_entry / (n_val * m_val)
    tmp_filtered = highpass_filter(np.fft.fftshift(tmp_result), n_val, m_val)
    tmp_filtered = np.fft.ifftshift(tmp_filtered)
    tmp_filtered = fourier_2d(tmp_filtered, n_val, m_val, True)
    tmp_filtered = np.abs(tmp_filtered)
    cv2.imwrite('./task3_result_dir' + '/' + file_list[idx][0:-4] + '_task3.jpg', tmp_filtered)
    tmp = 0
    for n_idx, entry in enumerate(tmp_filtered.tolist()):
        for m_idx, real_entry in enumerate(entry):
            if n_idx <= n_val * 0.05 or n_idx >= n_val * 0.95 and m_idx <= m_val * 0.05 or m_idx >= m_val * 0.95:
                continue
            else:
                tmp = tmp + real_entry
    tmp = tmp / (0.9 * n_val * 0.9 * m_val)
    if tmp >= 3.0:
        result = True
    else:
        result = False
    print(str(idx) + ': ' + str(result))
print("Task3 end")
