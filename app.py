import PySimpleGUI as psg
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import morphology
from skimage import measure
import math
import ffmpeg



def createPlot(file_list, file_distance):
    """
    Create a grpah plot
    """
    plt.plot(file_list, file_distance, color="blue")
    plt.title("Picking Frame Distance v.s. Frame Number", fontsize=14)
    plt.xlabel('Frame Number')
    plt.ylabel('Frame Distance')
    return plt.gcf()


def drawFigure(canvas, figure):
    """
    Draw the figure after the computation complete
    """
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def scoring(frame, list):
    """
    Score Fuction give a score of the current frame's edge based on the standard red line
    The more overlaping line area, the higher score
    The score is set to be positive (>0, good frames usually have score between 1300-2500)
    The function could be later modified to adjust the sensitivity to the dash line/ solid line
    by changing the min_line_length and max_line_gap

    :param frame: current frame in opencv format
    :param list: the red line coordinates
    """
    low_threshold = 50
    high_threshold = 150
    kernel_size = 5
    blur = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

    # Compute edge
    edges2 = cv2.Canny(blur[(list[3][1]-30):(list[3][1]+30),
                       list[3][0]:list[2][0]], low_threshold, high_threshold)
    edges1 = cv2.Canny(blur[(list[1][1]-30):(list[0][1]+30),
                       list[1][0]:list[0][0]], low_threshold, high_threshold)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # minimum number of votes (intersections in Hough grid cell)
    threshold = 15
    min_line_length = 100  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments
    # creating a blank to draw lines on
    line_image = np.zeros((120, list[2][0]-list[3][0], 3), dtype='uint8')

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines1 = cv2.HoughLinesP(edges1, rho, theta, threshold, np.array([]),
                             min_line_length, max_line_gap)
    lines2 = cv2.HoughLinesP(edges2, rho, theta, threshold, np.array([]),
                             min_line_length, max_line_gap)

    if (lines1 is not None):
        for line in lines1:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 15)

    if (lines2 is not None):
        for line in lines2:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1+60), (x2, y2+60), (255, 0, 0), 15)

    # create a standard line for scoring
    stand_line_image = np.zeros((120, list[2][0]-list[3][0], 3), dtype='uint8')
    stand_line_image = cv2.line(
        stand_line_image, (list[1][0]+30, 20), (list[0][0]-20, 20), (0, 255, 0), 15)
    stand_line_image = cv2.line(
        stand_line_image, (list[1][0]+30, 60), (list[0][0]-20, 60), (0, 255, 0), 15)
    stand_line_image = cv2.line(
        stand_line_image, (100, 30), (500, 30), (0, 255, 0), 15)
    stand_line_image = cv2.line(
        stand_line_image, (100, 90), (500, 90), (0, 255, 0), 15)

    # Compute score by awarding the overlapping area and punishing unreached area
    lines_edges_compare = cv2.addWeighted(
        line_image, 1, stand_line_image, 1, 0)
    score = 0
    n = 0
    for i in lines_edges_compare:
        for j in i:
            if ((j == np.array([255, 255, 0], dtype=np.uint8)).all()):
                score += 1
            # punish the area not within the standard line
            # disabled now
            # if (((j == [0,255,0]).all()) & n < 430 & n > 320):
            #         score -=1

    return score


def detectBlank(frame, color, sensitivity, list):
    """
    detectBlank Fuction returns true if detect the perforation, returns flase otherwise

    :param frame: current frame in opencv format
    :param color: 1: green 2: blue
    :param sensitivity: the tolerant area of the perforation color
    :param list: the coordinates of the standard red line
    """

    f = frame[(list[0][1])+30:(list[3][1])-30, list[1][0]+30:list[0][0]-30]
    if color == 1:
        fcomb = f[..., 1] >= 250
    elif color == 2:
        fcomb = f[..., 2] >= 250
    fcomb_open = morphology.binary_opening(
        fcomb.astype('uint8'), footprint=np.ones((5, 5)))
    all_labels = measure.label(fcomb_open, connectivity=2)
    np.unique(all_labels.flatten())
    howmany = len(np.unique(all_labels.flatten()))
    for i in range(1, howmany):
        if np.sum(all_labels == i) > sensitivity:
            return True
    return False


def detect_red(frame):
    """
    detect_red Fuction returns a list of coordinates of the four red lines in the video

    :param frame: current frame in opencv format
    :param color: 1: green 2: blue
    :param sensitivity: the tolerant area of the perforation color
    :param list: the coordinates of the standard red line
    """
    # threshold on red color
    lowcolor = (0, 0, 75)
    highcolor = (55, 55, 255)
    thresh = cv2.inRange(frame, lowcolor, highcolor)

    # apply morphology close
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # get contours and filter on area
    result = []
    contours = cv2.findContours(
         thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    width = 0
    for c in contours:
            area = cv2.contourArea(c)
            if area > 3000:
                x, y, w, h = cv2.boundingRect(c)
                width += w
                result.append([x, y])

    width = width/4
    result = sorted(result)
    result[0][0] = result[1][0]
    result[3][0] = result[2][0]
    result = sorted(result)
    bottom_right = result[3]
    top_right = result[2]
    bottom_left = result[1]
    bottom_left[0] += math.floor(width)
    top_left = result[0]
    top_left[0] += math.floor(width)

    return [top_right, top_left, bottom_right, bottom_left]


#---------------------------
# MAIN loop
# Create a app layout
#---------------------------
filmInput = [[psg.Text(text='Japanese Paper Film Digitalizer',
                       font=('Arial Bold', 20),
                       size=20,
                       expand_x=True,
                       justification='center')],
             [psg.Text('FILE', font=('Arial Bold', 15)), psg.In(size=(
                 25, 1), enable_events=True, key='-FILE-'), psg.FileBrowse()],  # input 1080p version file
             [psg.Text('FILE in 4k', font=('Arial Bold', 15)), psg.In(size=(
                 25, 1), enable_events=True, key='-FILE4k-'), psg.FileBrowse()],  # input higher resolution file
             [psg.Text('Frame frequency', font=('Arial Bold', 15)), psg.In(
                 size=(25, 1), enable_events=True, key='-FRAMEFEQ-')],  # input the frequency of picking good frames should be within 40-60
             [psg.Text('Starting frame number', font=('Arial Bold', 15)), psg.In(size=(25, 1), enable_events=True,
                                                                                 key='-FRAMENUM-')],  # the starting frame number should not exceed the frame number of the film
             [psg.Text('Perforation color', font=('Arial Bold', 15)),  # select the perforation color green/blue
              psg.Radio('Green Background', "RADIO1",
                        default=1, key="-COLOR-"),
              psg.Radio('Blue Background', "RADIO1", default=2, key="-COLOR-")],
             [psg.Text('Adjust Sensitivity of Detecting Perforation Area', font=('Arial Bold', 15)), psg.Slider(range=(0, 10), default_value=5,  # change the sensitiveity of the perforation, larger level means more tolerence to the larger area of perforation color
                                                                                                                expand_x=False, enable_events=True,
                                                                                                                orientation='horizontal', key='-Sensitivity-')],
             [psg.Text('Download Folder', font=('Arial Bold', 15)), psg.In(size=(
                 25, 1), enable_events=True, key='-FOLDER-'), psg.FolderBrowse()],  # select the destination folder
    [psg.OK(), psg.Cancel()]
]

graph = [[psg.Canvas(size=(1000, 1000), key='-CANVAS-')]]

layout = [[psg.Column(filmInput, element_justification='left'),
           psg.Column(graph, element_justification='c')]]

window = psg.Window('Japanese Paper Film Digitalizer',
                    layout, size=(1080, 500))

while True:
    event, values = window.read()
    if event == psg.WIN_CLOSED or event == 'Exit':
        break
    if event == "OK":
        file = values['-FILE-']
        file4k = values['-FILE4k-']
        framefeq = int(values['-FRAMEFEQ-'])
        i = int(values['-FRAMENUM-'])  # frame number
        color = int(values['-COLOR-'])
        des = values['-FOLDER-']
        # the area of background color in negative result is from 1000 to 2000
        sens = values['-Sensitivity-']*100+1000

        cap = cv2.VideoCapture(file)
        cap4k = cv2.VideoCapture(file4k)

        j = 0  # counter
        max = 0
        file_list = []

        # find a random frame to extract the coordinates of red lines
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        success, image = cap.read()
        list = detect_red(image)

        print("read the video!")

        while (success):
            for j in range(framefeq-10):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i+j)
                success, image = cap.read()
                if not success:
                    break
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                print("I read the frame %d" % (i+j))
                if (detectBlank(image, color, sens, list) == False):
                    print("Detect no blank area!")
                    score = scoring(image, list)
                    if (score > max):
                        print("Find a potential good frame")
                        max = score
                        print(max)
                        max_f = i+j
            if not success:
                break
            if max != 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max_f)
                # cap4k.set(cv2.CAP_PROP_POS_FRAMES, max_f)
                success, image = cap.read()
                # success4k, img = cap4k.read()

                # the new implemented function for reading 6k videos
                # TODO fix the problem: the output format cannot be read, might not works for python??
                out, _ = (
                    ffmpeg
                    .input(file4k)
                    .filter('select', 'eq(n,{})'.format(max_f))
                    .output('{}/frame{}.png'.format(des, max_f), vframes=1, format='image2', vcodec='prores', update=1)
                    .run(capture_stdout=True)
                )
                print("save frame%d.jpg" % max_f)
                file_list.append(max_f)
                i = max_f+10
            else:
                print("The max val is %d" % max)
                i = i+framefeq
            max = 0
            max_f = 0

        file_distance = [y - x for x, y in zip(file_list, file_list[1:])]
        drawFigure(window['-CANVAS-'].TKCanvas,
                   createPlot(file_list, file_distance))

    if event == "Cancel":
        break

window.close()
