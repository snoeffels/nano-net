import base64
import os
import statistics
import tkinter
import tkinter.filedialog as filedialog
import traceback
from glob import glob
from os import listdir
from os.path import isfile, join
from statistics import mean

import eel
import imageio.v2 as iio
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas
import pandas as pd
import scipy.ndimage.morphology as ndi
import seaborn as sns
from openpyxl.utils.dataframe import dataframe_to_rows
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.morphology import closing
from skimage.segmentation import clear_border
from skimage.segmentation import watershed
from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score as cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

##############################################################
#                       Globals                              #
##############################################################
OUTPUT_PATH = ""

PATH_1 = ""
PATH_2 = ""
PATH_3 = ""
PATH_4 = ""

PATHS = []
FEATURES = []
COLORS = []

SAMPLE_SIZE = 0
ACCURACY = 0
ACCURACY_BALANCED = 0
OPTIMAL_NEIGHBORS = 0

testSize = 0.2
nEstimators = 1000
minSamplesSplit = 2
minSamplesLeaf = 1
maxDepth = 110
testSizeKnn = 0.2
nNeighbors = 3
nComponents = 2
perplexity = 20
nIter = 500

# ----------------------------------------------------------------


CANCEL_SAME_NDS = False
CANCEL_DIFFERENT_NDS = False
CONDI_LIST = []
PIXEL = 9.02  # enter pixels of your image (can check this in fiji)


# l=list of paths containing the images to be quantified and compared together
# condi_list=list of conditions corresponding to the paths in l
# order= list of conditions in the order you want them to appear in the plot


# stat( f1, f2) # get tsne, knn, boxplot between 2 conditions

# print(globals())
# print(locals())

##############################################################
#                       UI/Backend Communication             #
##############################################################


@eel.expose
def set_pixel(p):
    global PIXEL
    PIXEL = p
    print("PIXEL was set to: " + str(PIXEL))


@eel.expose
def set_parameters(parameters):
    global testSize, nEstimators, minSamplesLeaf, minSamplesSplit, maxDepth, testSizeKnn, nNeighbors,  perplexity, nIter
    testSize = parameters['testSize']
    nEstimators = parameters['nEstimators']
    minSamplesLeaf = parameters['minSamplesLeaf']
    minSamplesSplit = parameters['minSamplesSplit']
    maxDepth = parameters['maxDepth']
    testSizeKnn = parameters['testSizeKnn']
    nNeighbors = parameters['nNeighbors']
    perplexity = parameters['perplexity']
    nIter = parameters['nIter']
    print("parameters were set")


@eel.expose
def set_conditions_paths_colors(conditions, paths, colors):
    global CONDI_LIST, PATHS, COLORS, SAMPLE_SIZE
    CONDI_LIST = conditions
    PATHS = paths
    COLORS = colors
    print("this colours: ", COLORS)
    SAMPLE_SIZE = get_samplesize(PATHS)


    return SAMPLE_SIZE  # return sample size here


@eel.expose
def set_optimal_neighbors(value):
    global OPTIMAL_NEIGHBORS
    OPTIMAL_NEIGHBORS = value
    print("OPTIMAL_NEIGHBORS was set to: " + str(OPTIMAL_NEIGHBORS))


@eel.expose
def set_features(features):
    global FEATURES
    FEATURES = features
    print("FEATURES was set to: (" + str(len(features)) + ") " + str(features))


@eel.expose
def get_accuracy():
    global CONDI_LIST, PATHS
    optimal_k, accuracy, accuracy_balanced = semua(PATHS, CONDI_LIST, dry_run=True)
    return {
        "accuracy": accuracy,
        "neighbors": optimal_k,
    }


@eel.expose
def get_sample_size():
    global SAMPLE_SIZE
    SAMPLE_SIZE = get_samplesize(PATHS)

    return SAMPLE_SIZE


@eel.expose
def select_path_1():
    global PATH_1
    PATH_1 = select_folder_tk()
    print("PATH_1 was set to: " + PATH_1)
    return PATH_1


@eel.expose
def select_path_2():
    global PATH_2
    PATH_2 = select_folder_tk()
    print("PATH_2 was set to: " + PATH_2)
    return PATH_2


@eel.expose
def select_path_3():
    global PATH_3
    PATH_3 = select_folder_tk()
    print("PATH_3 was set to: " + PATH_3)
    return PATH_3


@eel.expose
def select_path_4():
    global PATH_4
    PATH_4 = select_folder_tk()
    print("PATH_4 was set to: " + PATH_4)
    return PATH_4


@eel.expose
def select_output_path():
    global OUTPUT_PATH
    OUTPUT_PATH = select_folder_tk()
    print("OUTPUT_PATH was set to: " + OUTPUT_PATH)
    return OUTPUT_PATH


def select_folder_tk():
    root = tkinter.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    # TODO handle cancel event
    return filedialog.askdirectory()


@eel.expose
def run_same_nds():
    one_condition_workflow(PATH_1)


@eel.expose
def cancel_same_nds():
    global CANCEL_SAME_NDS
    CANCEL_SAME_NDS = True
    print("CANCEL_SAME_NDS was set to: " + str(CANCEL_SAME_NDS))


@eel.expose
def run_different_nds():
    global PATHS, CONDI_LIST
    try:
        semua(PATHS, CONDI_LIST, False)
    except Exception:
        traceback.print_exc()


@eel.expose
def cancel_different_nds():
    global CANCEL_DIFFERENT_NDS
    CANCEL_DIFFERENT_NDS = True
    print("CANCEL_DIFFERENT_NDS was set to: " + str(CANCEL_DIFFERENT_NDS))


##############################################################
#                       Actual Logic                         #
##############################################################
def get_samplesize(PATHS): 
    samplesize_list=[]
    for countfolder in PATHS:
        onlyfiles = [f for f in listdir(countfolder) if isfile(join(countfolder, f))]
        for i in onlyfiles:
            if i.endswith(".tif"):
                samplesize_list.append(i)
    samplesize = len(samplesize_list)
    print(len(samplesize_list))
    print(samplesize_list)
    print(samplesize)
    return samplesize

                




    

# get excel and pictures:
def one_condition_workflow(path1):
    global CANCEL_SAME_NDS
    CANCEL_SAME_NDS = False
    
    # to fix changing plot style bug we need to set those properties for every execution
    sns.set_style('white')
    
    p = []
    names = []
    onlyfiles = [f for f in listdir(path1) if isfile(join(path1, f))]
    for i in onlyfiles:
        if i.endswith(".tif"):
            names.append(i)
            path = os.path.join(mypath, i)
            p.append(path)

    count = 0
    seg = 0

    for i in range(len(p)):
        eel.sleep(0.001)

        if CANCEL_SAME_NDS:
            break

        count += 1
        seg += 1
        name = names[i]
        b = glob(p[i])
        image = iio.imread(b[0])
        props4, arry_im, list_size4, mean_area, var_size, density, list_fluo, list_relative_fluo, mean_fluo, var_fluo, list_intensity_max, mean_intensity_max, list_intensity_min, mean_intensity_min, list_area_filled, mean_area_filled, list_axis_major_length, mean_axis_major_length, list_axis_minor_length, mean_axis_minor_length, list_eccentricity, mean_eccentricity, list_equivalent_diameter_area, mean_equivalent_diameter_area, list_perimeter, mean_perimeter, list_label, sum_label, image, thresh3, closed3, cleared3, image_label_overlay4, label_image4, sci, density_microns = mybin(
            image, PIXEL)

        pic(i, len(p), image, thresh3, closed3, cleared3, image_label_overlay4, label_image4, name, path1, seg)

        columns = ["name", "area", "mean_area", "var_area", "density", "intensity", "relative_intensity",
                   "mean_intensity", "var_intensity", "max_intensity", "mean_max_intensity", "min_intensity",
                   "mean_min_intensity", "area_filled", "mean_area_filled", "major_axis_length",
                   "mean_major_axis_length", "minor_axis_length", "mean_minor_axis_length", "eccentricity",
                   "mean_eccentricity", "equivalent_diameter_area", "mean_equivalent_diameter_area", "perimeter",
                   "mean_perimeter", "nano_domain_id", "nano_domain_quantity", "sci", "density_microns"]
        new = list_size4

        if len(list_size4) > 0:
            for j in range(len(list_size4)):
                new[j] = [name, list_size4[j], mean_area, var_size, density, list_fluo[j], list_relative_fluo[j],
                          mean_fluo, var_fluo, list_intensity_max[j], mean_intensity_max, list_intensity_min[j],
                          mean_intensity_min, list_area_filled[j], mean_area_filled, list_axis_major_length[j],
                          mean_axis_major_length, list_axis_minor_length[j], mean_axis_minor_length,
                          list_eccentricity[j], mean_eccentricity, list_equivalent_diameter_area[j],
                          mean_equivalent_diameter_area, list_perimeter[j], mean_perimeter, list_label[j], sum_label,
                          sci, density_microns]

            rows = new
            data = pd.DataFrame(rows, columns=columns)

        else:
            new = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            new[0][0] = name

            rows = new
            data = pd.DataFrame(rows, columns=columns)

        if 'file' in globals():
            if os.path.isfile(file) == True:  # if file already exists append to existing file
                workbook = openpyxl.load_workbook(file)  # load workbook if already exists
                sheet = workbook['Sheet1']  # declare the active sheet
                for row in dataframe_to_rows(data, header=False, index=False):
                    sheet.append(row)
                workbook.save(file)  # save workbook
                workbook.close()  # close workbook
        else:  # create the excel file if doesn't already exist

            if count == 1:

                out_path = OUTPUT_PATH + "/Test_results.xlsx"
                writer = pd.ExcelWriter(out_path, engine='xlsxwriter')
                data.to_excel(writer, sheet_name='Sheet1', header=True, index=False)
                writer.save()
                writer.close()
            else:
                workbook = openpyxl.load_workbook(out_path)  # load workbook if already exists
                sheet = workbook['Sheet1']  # declare the active sheet
                for row in dataframe_to_rows(data, header=False, index=False):
                    sheet.append(row)
                workbook.save(out_path)  # save workbook
                workbook.close()  # close workbook

    CANCEL_SAME_NDS = False

##################################################################################################################
# for visual pictures- works
def pic(i, p, image, thresh3, closed3, cleared3, image_label_overlay4, label_image4, name, path1, seg):
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 10), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('image', fontsize=20)

    ax[1].imshow(thresh3, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('thresh3', fontsize=20)

    ax[2].imshow(closed3)
    ax[2].axis('off')
    ax[2].set_title('closed3', fontsize=20)

    ax[3].imshow(cleared3)
    ax[3].axis('off')
    ax[3].set_title('cleared3', fontsize=20)

    ax[4].imshow(image_label_overlay4)
    for region in regionprops(label_image4):
        # take regions with large enough areas
        if region.area >= 1:
            # draw rectangle around segmented ND
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
            ax[4].add_patch(rect)

    ax[4].axis('off')
    ax[4].set_title('rect_w', fontsize=20)

    fig.tight_layout()

    pltPath = os.path.join(OUTPUT_PATH, name) + str(seg) + ".png"
    plt.savefig(pltPath, format='png')
    plt.close(fig)

    tmp = open(pltPath, "rb")
    eel.add_image_same_nd(
        {'src': base64.b64encode(tmp.read()).decode('utf-8'), 'title': str(name) + str(seg) + ".png"}, p, i + 1)
    tmp.close()


##################################################################### for many statistics:
def semua(l, order, dry_run=False):
    
    

    # to fix changing plot style bug we need to set those properties for every execution
    sns.set_style('white')

    df_all = pd.DataFrame()
    df_list = [0] * len(l)
    names_all = [0] * len(l)
    arry_l = [0] * len(l)
    laenge = [0] * len(l)

    for num, kata in enumerate(l):
        df_list[num], names_all[num], arry_l[num] = paths_plot(kata)

        # df_all = df_all.append(df_list[num], ignore_index=True)
        df_all = pandas.concat((df_all, df_list[num]), ignore_index=True)

        laenge[num] = df_list[num].shape[0]

    for i in range(len(arry_l)):
        if i == 0:
            if len(arry_l) == 3:
                arry_all = np.concatenate((arry_l[i], arry_l[i + 1], arry_l[i + 2]), axis=0)
            elif len(arry_l) == 2:
                arry_all = np.concatenate((arry_l[i], arry_l[i + 1]), axis=0)
            elif len(arry_l) == 4:
                arry_all = np.concatenate((arry_l[i], arry_l[i + 1], arry_l[i + 2], arry_l[i + 3]), axis=0)

    condition = 1 * names_all
    condition2 = []
    names_all2 = []
    

    for i, j in enumerate(CONDI_LIST):
        condition[i] = [j] * len(names_all[i])

    for i, j in enumerate(names_all):
        for index, name in enumerate(j):
            names_all2.append(name)

    for i, j in enumerate(condition):
        for index, name in enumerate(j):
            condition2.append(name)

    condi_df = []
    for i, j in enumerate(CONDI_LIST):
        condi_df.append(laenge[i] * [j])

    condi_df2 = []
    for i, j in enumerate(condi_df):
        for index, name in enumerate(j):
            condi_df2.append(name)

    df_all["condition"] = condi_df2
    
    
  

    # calculate accuracy
    if dry_run:
        return knn_all(arry_all, condition2, order, True)

    plots_all(df_all, order)

    plot_correlation(df_all)

    tsn_all(arry_all, names_all2, condition2)

    knn_all(arry_all, condition2, order)

    forest(arry_all, condition2, order)

    return arry_all, names_all2, condition2


#######################################################################################
def tsn_all(arry_all, names_all2, condition2):  # tnse

    scaler = preprocessing.StandardScaler().fit(arry_all)
    X_scaled = scaler.transform(arry_all)

    tsne = []

    global perplexity, nIter
    
    print("perplexity -> " + str(perplexity))
    print("nIter -> " + str(nIter))

    try:
        tsne = TSNE(n_components=2, perplexity=int(perplexity), n_iter=int(nIter)).fit_transform(
            X_scaled)
    except ValueError:
        # TODO Fix this!
        print("perplexity must be less than n_samples")

    x = tsne[:, 0]
    y = tsne[:, 1]

    df = pd.DataFrame(dict(x=x, y=y, label=condition2, img_name=names_all2))
    df.groupby('label')

    sns.scatterplot(x="x", y="y", hue="label", data=df, palette=COLORS)

    global OUTPUT_PATH
    pltPath = os.path.join(OUTPUT_PATH, "tsne") + ".png"

    plt.savefig(pltPath, format='png')
    plt.close()

    tmp = open(pltPath, "rb")
    eel.add_image_different_nd(
        {'src': base64.b64encode(tmp.read()).decode('utf-8'), 'title': "tsne.png"}, 1, 1, "tsne")
    tmp.close()

    for i in range(df.shape[0]):
        plt.text(x=df.x[i] + 0.3, y=df.y[i] + 0.3, s=df.img_name[i])

    return arry_all, names_all2, condition2


########################################################################
def plot_correlation(df_all):
    corr = df_all.corr()
    f, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=plt.cm.RdBu,
                square=True, ax=ax)  # cmap=sns.diverging_palette(220, 10, as_cmap=True)

    pltPath = os.path.join(OUTPUT_PATH, "correlation") + ".png"
    plt.savefig(pltPath, format='png')
    plt.cla()
    plt.clf()
    plt.close()

    tmp = open(pltPath, "rb")
    eel.add_image_different_nd(
        {'src': base64.b64encode(tmp.read()).decode('utf-8'), 'title': "correlation.png"},
        1, 1,
        "boxplot")
    tmp.close()

    return df_all


########################################################################
def knn_all(arry_all, condition2, order, dry_run=False):
    global ACCURACY, ACCURACY_BALANCED, OPTIMAL_NEIGHBORS, nNeighbors, testSizeKnn
    condition2 = np.array(condition2)
    MinMaxScaler = preprocessing.MinMaxScaler()
    arry_norm = MinMaxScaler.fit_transform(arry_all)

    X_train, X_test, y_train, y_test = train_test_split(arry_norm, condition2, test_size=float(testSizeKnn))

    nNeighbors = int(nNeighbors)
    neighbors = list(range(1, nNeighbors + 1, 1))

    print(neighbors)
    # empty list that will hold cv scores
    cv_scores = []

    # perform 10-fold cross validation
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=int(k))
        scores = cross_val_score(knn, X_train, y_train, cv=8, scoring='accuracy')
        cv_scores.append(scores.mean())

    # changing to misclassification error
    mse = [1 - x for x in cv_scores]

    # determining best k
    optimal_k = neighbors[mse.index(min(mse))]
    print("The optimal number of neighbors is {}".format(optimal_k))
    model = KNeighborsClassifier(n_neighbors=optimal_k)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    accuracy_balanced = balanced_accuracy_score(y_test, predicted)

    print(predicted)
    print(y_test)
    print("accuracy: {}".format(accuracy))
    print("accuracy_balanced: {}".format(accuracy_balanced))

    if dry_run:
        ACCURACY = accuracy
        ACCURACY_BALANCED = accuracy_balanced
        OPTIMAL_NEIGHBORS = optimal_k
        return optimal_k, accuracy, accuracy_balanced

    fig_dims = (6, 6)
    plt.subplots(figsize=fig_dims)
    sns.scatterplot(X_test[:, 0], X_test[:, 6], hue=predicted, hue_order=order, s=50, palette=COLORS)

    pltPath = os.path.join(OUTPUT_PATH, "knn_predicted") + ".png"
    plt.savefig(pltPath, format='png')
    plt.cla()
    plt.clf()
    plt.close()

    tmp = open(pltPath, "rb")
    eel.add_image_different_nd(
        {'src': base64.b64encode(tmp.read()).decode('utf-8'), 'title': "knn_predicted.png"}, 2, 1, "knn")
    tmp.close()

    fig_dims = (6, 6)
    plt.subplots(figsize=fig_dims)
    sns.scatterplot(X_test[:, 0], X_test[:, 6], hue=y_test, hue_order=order,  s=50, palette=COLORS)

    pltPath = os.path.join(OUTPUT_PATH, "knn_true") + ".png"
    plt.savefig(pltPath, format='png')
    plt.cla()
    plt.clf()
    plt.close()

    tmp = open(pltPath, "rb")
    eel.add_image_different_nd(
        {'src': base64.b64encode(tmp.read()).decode('utf-8'), 'title': "knn_true.png"}, 2, 2, "knn")
    tmp.close()

    return arry_all, condition2


########################################################################
def forest(arry_all, condition2, order):
    global testSize, nEstimators, minSamplesSplit, minSamplesLeaf, maxDepth

    arr_col = ["me", "var", "mean_blur", "var_blur", "mean_fluo", "var_fluo", "mean_size", "var_size", "mean_int_max",
               "mean_int_min", "area_filled", "major_axis", "minor_axis", "eccentricity", "equivalent_diameter",
               "perimeter", "nano_domain_quantity", "density", "relative_fluo", "density_microns"]

    condition2 = np.array(condition2)

    X_train, X_test, y_train, y_test = train_test_split(arry_all, condition2, test_size=float(testSize),
                                                        random_state=12345, stratify=condition2)

    global nEstimators, minSamplesSplit, minSamplesLeaf, maxDepth
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=nEstimators, random_state=12345, min_samples_split=minSamplesSplit,
                                 min_samples_leaf=minSamplesLeaf, max_features="sqrt", max_depth=maxDepth,
                                 bootstrap=True)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    eel.print_to_report("Accuracy of Forest_1: " + str(accuracy_score(y_test, y_pred)))
    # print("Accuracy of Forest_1: ", str(accuracy_score(y_test, y_pred)))

    feature_imp = pd.Series(clf.feature_importances_, index=arr_col).sort_values(ascending=False)
    eel.print_to_report(str(feature_imp))
    # print(feature_imp)

    # View the classification report for test data and predictions
    eel.print_to_report(str(classification_report(y_test, y_pred)))
    # print(classification_report(y_test, y_pred))

    # Get and reshape confusion matrix data
    matrix = confusion_matrix(y_test, y_pred)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    # Build the plot
    plt.figure(figsize=(16, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size': 10},
                cmap=plt.cm.RdBu, linewidths=0.2)  # cmap=plt.cm.Greens

    # Add labels to the plot
    class_names = order
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Random Forest Model')

    pltPath = os.path.join(OUTPUT_PATH, "forest-confusion-matrix") + ".png"
    plt.savefig(pltPath, format='png')

    tmp = open(pltPath, "rb")
    eel.add_image_different_nd(
        {'src': base64.b64encode(tmp.read()).decode('utf-8'), 'title': "forest-confusion-matrix.png"},
        2, 1,
        "rf")
    tmp.close()

    ### tree that works: plot of the decision tree
    fn = arr_col
    cn = condition2
    plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
    tree.plot_tree(clf.estimators_[0],
                   feature_names=fn,
                   class_names=cn,
                   filled=True)

    pltPath = os.path.join(OUTPUT_PATH, "forest-decision-tree") + ".png"
    plt.savefig(pltPath, format='png')

    tmp = open(pltPath, "rb")
    eel.add_image_different_nd(
        {'src': base64.b64encode(tmp.read()).decode('utf-8'), 'title': "forest-decision-tree.png"},
        2, 2,
        "rf")
    tmp.close()

    plt.cla()
    plt.clf()
    plt.close()


########################################################################
def plots_all(df_all, order):  # boxplots
    global OUTPUT_PATH, FEATURES
    lookup = {
        "area": {
            "title": "ND area (microns)",
            "ylabel": "microns"
        },

        "mean_area": {
            "title": "mean ND area per image",
            "ylabel": "microns"
        },

        "var_area": {
            "title": "variance of ND area per image",
            "ylabel": "microns"
        },

        "density": {
            "title": "ND density",
            "ylabel": "percent"
        },

        "intensity": {
            "title": "ND intensity",
            "ylabel": "intensity"
        },

        "relative_intensity": {
            "title": "relative ND intensity",
            "ylabel": "rel. intensity"
        },

        "mean_intensity": {
            "title": "mean ND intensity per image",
            "ylabel": "mean intensity"
        },

        "var_intensity": {
            "title": "variance of ND intensity per image",
            "ylabel": "intensity variance"
        },

        "max_intensity": {
            "title": "ND max. intensity",
            "ylabel": "max. intensity"
        },

        "mean_eccentricity": {
            "title": "mean ND eccentricity per image",
            "ylabel": "eccentricity"
        },

        "equivalent_diameter_area": {
            "title": "ND equivalent diameter area",
            "ylabel": "microns"
        },

        "mean_equivalent_diameter_area": {
            "title": "mean ND equivalent diameter area per image",
            "ylabel": "microns"
        },

        "perimeter": {
            "title": "perimeter of NDs",
            "ylabel": "microns"
        },

        "mean_perimeter": {
            "title": "mean perimeter of NDs per image",
            "ylabel": "microns"
        },

        "nano_domain_quantity": {
            "title": "Nr. of NDs per image",
            "ylabel": "ND quantitiy"
        },

        "sci": {
            "title": "image wide spatial clustering index",
            "ylabel": "SCI"
        },

        "density_microns": {
            "title": "ND density in microns",
            "ylabel": "microns"
        }
    }

    # print(df_all)
    df_unique=df_all.copy(deep=True)
    df_unique = df_unique.drop_duplicates(subset=['name'])
    print(df_unique.columns)
    for i, feature in enumerate(FEATURES):
        plt.subplot(1, 1, 1)

        # sns.boxplot(x="condition", y="density", data=df_all, order=order, palette="Paired", showfliers=False)
        sns.boxplot(x="condition", y=feature, data=df_unique, palette=COLORS, showfliers=False)
        sns.swarmplot(x="condition", y=feature, data=df_unique,  color="grey", alpha=1, size=4)

        plt.title(lookup[feature]['title'])
        plt.xlabel('condition')
        plt.ylabel(lookup[feature]['ylabel'])

        # plt.subplots_adjust(bottom=0.1, right=4, top=1.4)

        pltPath = os.path.join(OUTPUT_PATH, "boxplot-" + str(feature)) + ".png"
        plt.savefig(pltPath, format='png')
        plt.cla()
        plt.clf()
        plt.close()

        tmp = open(pltPath, "rb")
        eel.add_image_different_nd(
            {'src': base64.b64encode(tmp.read()).decode('utf-8'), 'title': "boxplot-" + str(feature) + ".png"},
            len(FEATURES), (i + 1),
            "boxplot")
        tmp.close()

    return df_all


########################################################################
def paths_plot(mypath):
    p = []
    names = []
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for i in onlyfiles:
        if i.endswith(".tif"):
            names.append(i)
            path = os.path.join(mypath, i)
            p.append(path)
    arry = np.ones((len(names), 20))
    df = pd.DataFrame()

    for i in range(len(p)):

        name = names[i]
        b = glob(p[i])
        image = iio.imread(b[0])

        props4, arry_im, list_size4, mean_area, var_size, density, list_fluo, list_relative_fluo, mean_fluo, var_fluo, list_intensity_max, mean_intensity_max, list_intensity_min, mean_intensity_min, list_area_filled, mean_area_filled, list_axis_major_length, mean_axis_major_length, list_axis_minor_length, mean_axis_minor_length, list_eccentricity, mean_eccentricity, list_equivalent_diameter_area, mean_equivalent_diameter_area, list_perimeter, mean_perimeter, list_label, sum_label, image, thresh3, closed3, cleared3, image_label_overlay4, label_image4, sci, density_microns = mybin(
            image, PIXEL)
        # print(arry_im)
        columns = ["name", "area", "mean_area", "var_area", "density", "intensity", "relative_intensity",
                   "mean_intensity", "var_intensity", "max_intensity", "mean_max_intensity", "min_intensity",
                   "mean_min_intensity", "area_filled", "mean_area_filled", "major_axis_length",
                   "mean_major_axis_length", "minor_axis_length", "mean_minor_axis_length", "eccentricity",
                   "mean_eccentricity", "equivalent_diameter_area", "mean_equivalent_diameter_area", "perimeter",
                   "mean_perimeter", "nano_domain_id", "nano_domain_quantity", "sci", "density_microns"]
        new = list_size4
        if len(list_size4) > 0:
            for j in range(len(list_size4)):
                new[j] = [name, list_size4[j], mean_area, var_size, density, list_fluo[j], list_relative_fluo[j],
                          mean_fluo, var_fluo, list_intensity_max[j], mean_intensity_max, list_intensity_min[j],
                          mean_intensity_min, list_area_filled[j], mean_area_filled, list_axis_major_length[j],
                          mean_axis_major_length, list_axis_minor_length[j], mean_axis_minor_length,
                          list_eccentricity[j], mean_eccentricity, list_equivalent_diameter_area[j],
                          mean_equivalent_diameter_area, list_perimeter[j], mean_perimeter, list_label[j], sum_label,
                          sci, density_microns]
            rows = new
            data = pd.DataFrame(rows, columns=columns)

        else:
            new = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            new[0][0] = name

            print("no data")

            rows = new
            data = pd.DataFrame(rows, columns=columns)

        # df = df.append(data, ignore_index=True)
        df = pandas.concat((df, data), ignore_index=True)

        arry[i] = arry_im

    return (df, names, arry)


###################################################################################################
def calculate_sci(bild):
    intensity = np.asarray(bild)
    lys = intensity.tolist()
    flat = [val for sublist in lys for val in sublist]
    for i in flat:
        if i == 0:
            i = 0.001

    flat.sort()
    l = len(flat)

    low5 = int(l * 10 / 100)

    high5 = l - low5

    lowarry = flat[:low5]
    higharry = flat[high5:]

    mean1 = np.mean(lowarry)
    mean2 = np.mean(higharry)
    ratio = mean2 / mean1
    return ratio


##############################################################################################
# estimation of parameter lambda of a poisson distribution- works


def mybin(image, pixel):
    # pixelsize= (9.02/image.shape[0])**2 # 1 pixel has 0.0451 microns-> need squared
    # pixellength=9.02/image.shape[0]

    pixel = float(pixel)
    pixelsize = (pixel / image.shape[0]) ** 2  # enter the pixelsize in microns
    pixellength = pixel / image.shape[0]  # enter the pixelsize in microns
    sci = calculate_sci(image)

    me = image.mean()
    ratio = 2 * me
    blurred = gaussian_filter(image, sigma=0.9)

    thresh3 = 1 * blurred
    thresh3[thresh3 < ratio] = 0
    thresh3[thresh3 > ratio] = 1

    closed3 = closing(thresh3)  # dilation followed by an erosion
    cleared3 = clear_border(closed3)
    distance3 = ndi.distance_transform_edt(cleared3)
    coords3 = peak_local_max(distance3, labels=cleared3, footprint=np.ones((5, 5)), min_distance=8, exclude_border=True)
    mask3 = np.zeros(distance3.shape, dtype=bool)
    mask3[tuple(coords3.T)] = True
    markers3, _ = ndi.label(mask3)
    labels_water3 = watershed(-distance3, markers3, mask=cleared3, watershed_line=True)
    label_image4 = label(labels_water3)
    image_label_overlay4 = label2rgb(label_image4, image=image, bg_label=0, alpha=1)

    props4 = regionprops(label_image=label_image4, intensity_image=image, cache=True)

    list_size4 = []  # for watershed
    for i in range(len(props4)):
        if props4[i].area >= 1:
            list_size4.append(props4[i].area * pixelsize)
            # print("area", props4[i].area *pixelsize)
    if len(list_size4) > 0:
        mean_area = mean(list_size4) * pixelsize
    else:
        mean_area = 0

    if len(list_size4) > 0:
        density = ((sum(list_size4) * pixelsize) / image.shape[0])
    else:
        density = 0

    list_fluo = []
    list_relative_fluo = []
    for i in range(len(props4)):
        if props4[i].area >= 1:
            list_fluo.append(props4[i].mean_intensity)
            list_relative_fluo.append((props4[i].mean_intensity) / me)
    if len(list_size4) > 0:
        mean_fluo = mean(list_fluo)
        mean_relative_fluo = mean(list_relative_fluo)
    else:
        mean_fluo = 0
        mean_relative_fluo = 0

    list_intensity_max = []
    for i in range(len(props4)):
        if props4[i].area >= 1:
            list_intensity_max.append(props4[i].max_intensity)
    if len(list_size4) > 0:
        mean_intensity_max = mean(list_intensity_max)
    else:
        mean_intensity_max = 0

    list_intensity_min = []
    for i in range(len(props4)):
        if props4[i].area >= 1:
            list_intensity_min.append(props4[i].min_intensity)
    if len(list_size4) > 0:
        mean_intensity_min = mean(list_intensity_min)
    else:
        mean_intensity_min = 0

    list_area_filled = []
    for i in range(len(props4)):
        if props4[i].area >= 1:
            list_area_filled.append(props4[i].filled_area * pixelsize)
    if len(list_size4) > 0:
        mean_area_filled = mean(list_area_filled) * pixelsize
    else:
        mean_area_filled = 0

    list_axis_major_length = []
    for i in range(len(props4)):
        if props4[i].area >= 1:
            list_axis_major_length.append(props4[i].major_axis_length * pixellength)
    if len(list_size4) > 0:
        mean_axis_major_length = mean(list_axis_major_length) * pixellength
    else:
        mean_axis_major_length = 0

    list_axis_minor_length = []
    for i in range(len(props4)):
        if props4[i].area >= 1:
            list_axis_minor_length.append(props4[i].minor_axis_length * pixellength)
    if len(list_size4) > 0:
        mean_axis_minor_length = mean(list_axis_minor_length) * pixellength
    else:
        mean_axis_minor_length = 0

    list_eccentricity = []
    for i in range(len(props4)):
        if props4[i].area >= 1:
            list_eccentricity.append(props4[i].eccentricity)

    if len(list_size4) > 0:
        mean_eccentricity = mean(list_eccentricity)
    else:
        mean_eccentricity = 0

    list_equivalent_diameter_area = []
    for i in range(len(props4)):
        if props4[i].area >= 1:
            list_equivalent_diameter_area.append(props4[i].equivalent_diameter * pixellength)
    if len(list_size4) > 0:
        mean_equivalent_diameter_area = mean(list_equivalent_diameter_area) * pixellength
    else:
        mean_equivalent_diameter_area = 0

    list_perimeter = []
    for i in range(len(props4)):
        if props4[i].area >= 1:
            list_perimeter.append(props4[i].perimeter * pixellength)

    if len(list_size4) > 0:
        mean_perimeter = mean(list_perimeter) * pixellength
    else:
        mean_perimeter = 0

    list_label = []
    for i in range(len(props4)):
        if props4[i].area >= 1:
            list_label.append(props4[i].label)
    if len(list_size4) > 0:
        sum_label = len(list_label)

    else:
        sum_label = 0

    if len(list_size4) > 0:
        var_fluo = 2 * (statistics.pstdev(list_fluo))
        var_size = 2 * (statistics.pstdev(list_size4))
        density_microns = sum_label / pixel
    else:
        var_fluo = 0
        var_size = 0
        density_microns = 0

    arry_im = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    arry_im[0] = me
    arry_im[1] = np.var(image)
    arry_im[2] = blurred.mean()
    arry_im[3] = np.var(blurred)
    arry_im[4] = mean_fluo
    arry_im[5] = var_fluo
    arry_im[6] = mean_area
    arry_im[7] = var_size
    arry_im[8] = mean_intensity_max
    arry_im[9] = mean_intensity_min
    arry_im[10] = mean_area_filled
    arry_im[11] = mean_axis_major_length
    arry_im[12] = mean_axis_minor_length
    arry_im[13] = mean_eccentricity
    arry_im[14] = mean_equivalent_diameter_area
    arry_im[15] = mean_perimeter
    arry_im[16] = sum_label
    arry_im[17] = density
    arry_im[18] = mean_relative_fluo
    arry_im[19] = density_microns

    return (
        props4, arry_im, list_size4, mean_area, var_size, density, list_fluo, list_relative_fluo, mean_fluo, var_fluo,
        list_intensity_max, mean_intensity_max, list_intensity_min, mean_intensity_min, list_area_filled,
        mean_area_filled,
        list_axis_major_length, mean_axis_major_length, list_axis_minor_length, mean_axis_minor_length,
        list_eccentricity,
        mean_eccentricity, list_equivalent_diameter_area, mean_equivalent_diameter_area, list_perimeter, mean_perimeter,
        list_label, sum_label, image, thresh3, closed3, cleared3, image_label_overlay4, label_image4, sci,
        density_microns)


eel.init('dist')
eel.start('index.html', size=(900, 900), port=8080)
