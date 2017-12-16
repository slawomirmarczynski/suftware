from flask import Flask, render_template, request, session

#from io import BytesIO
import io
import base64

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

import uuid
import shutil

from deft_1d.deft_code import deft_1d
from deft_1d.sim import simulate_data_1d

# name of the flask app
app = Flask(__name__)

# this key decrypts the cookie on the client's browser
app.secret_key = os.urandom(32)


# deft home page
@app.route('/', methods=['GET','POST'])
def index():

    # dictionary to store all metadata in key value pairs
    metaData = {}

    # create temp files like metadata
    if 'uid' in session:
        print("Temp file already created")
        metaDataFile = str(session['uid']) + ".meta"
    else:
        session['uid'] = uuid.uuid4()
        metaDataFile = str(session['uid']) + ".meta"
        print("New temp file created")

    sample_distribution_type = ["gaussian", "narrow", "wide", "foothills", "accordian", "goalposts", "towers","uniform",
                                "beta_convex", "beta_concave", "exponential", "gamma", "triangular", "laplace","vonmises"]

    # example data list
    example_data = ['buffalo_snowfall.dat','old_faithful_eruption_times.dat','old_faithful_waiting_times.dat','treatment_length.dat']
    example_data_dict = {}
    example_data_selected = 'buffalo_snowfall.dat'

    # write following parameters to metadata files

    # default N value
    N = 100
    data_type = 'wide'
    # Deft parameter settings
    G = 100
    alpha = 3
    bbox_left = -6
    bbox_right = 6
    bbox = [bbox_left,bbox_right]

    # input_type default value is simulated, user post value will be written to
    # metadata file and update in the html from there.
    input_type = 'simulated'

    # simulate_index that selects default value of distribution list
    simulate_index = 0
    # similar index but for example data
    example_data_index = 0

    # list of example data

    example_data_dict['buffalo_snowfall.dat'] = np.loadtxt('./data/buffalo_snowfall.dat').astype(np.float64)
    example_data_dict['old_faithful_eruption_times.dat'] = np.loadtxt('./data/old_faithful_eruption_times.dat').astype(np.float)
    example_data_dict['old_faithful_waiting_times.dat'] = np.loadtxt('./data/old_faithful_waiting_times.dat').astype(np.float)
    example_data_dict['treatment_length.dat'] = np.loadtxt('./data/treatment_length.dat').astype(np.float)
    '''
    buffalo_snowfall = np.load('./data/'+str(example_data[0]))
    old_faithful_eruption_times = np.load('./data/' + str(example_data[1]))
    old_faithful_waiting_times = np.load('./data/' + str(example_data[2]))
    treatment_length = np.load('./data/' + str(example_data[3]))
    '''

    if request.method == 'POST':

        if str(request.form.get("run_deft_button")) == 'run_deft':

            # update deft parameters via post
            N = int(request.form['N'])
            G = int(request.form['G'])
            alpha = int(request.form['alpha'])
            bbox_left = int(request.form['bbox_left'])
            bbox_right = int(request.form['bbox_right'])
            bbox = [int(request.form['bbox_left']),int(request.form['bbox_right'])]

            # read radio button value
            if request.form.getlist('input_type'):
                input_type = str(request.form['input_type'])
                print(" Radio button value: ",input_type)
                metaData['input_type'] = input_type

            # write metadata to file
            with open(metaDataFile, "w") as myfile:
                for key in sorted(metaData):
                    myfile.write(key + ":" + str("".join(metaData[key])) + "\n")

            data_type = request.form['distribution']
            example_data_selected = str(request.form['example_data'])

            # the following two loops and their indices need to be replaced
            # by dicts:

            # loop for selecting the default value of distribution select
            for dist in sample_distribution_type:
                if str(data_type) == dist:
                    break
                simulate_index += 1

            # loop for selecting the default value of example_data select
            for example in example_data:
                if str(example_data_selected) == example:
                    break
                example_data_index += 1

    data, defaults = simulate_data_1d.run(data_type, N)
    #print(example_data_dict['buffalo_snowfall.dat'])
    loaded_data = example_data_dict[example_data_selected]
    #print(loaded_data)

    '''
    if input_type == 'simulated':
        # Simulate data and get default deft settings
        data, defaults = simulate_data_1d.run(data_type, N)
    elif input_type == 'example':
        # load example data here:
        pass
    '''

    # read any meta data written to file
    # check if file exists first
    read_metadata_dict = {}
    try:
        with open(metaDataFile) as myfile:
            for line in myfile:
                name, var = line.partition(":")[::2]
                read_metadata_dict[name.strip()] = var
    except FileNotFoundError:
        print("No metadata file")

    # read radio button value
    if 'input_type' in read_metadata_dict:
        input_type = str(read_metadata_dict['input_type']).strip()

    if input_type == 'simulated':
        # Do density estimation
        results = deft_1d.run(data, G=G, alpha=alpha, bbox=bbox, periodic=False, num_samples=0, print_t=False, tollerance=1E-3)
    elif input_type == 'example':
        results = deft_1d.run(loaded_data, G=G, alpha=alpha, bbox=bbox, periodic=False, num_samples=0, print_t=False,tollerance=1E-3)

    # Compute true density
    xs = results.bin_centers
    Q_true = np.zeros(G)
    for i, x in enumerate(xs):
        Q_true[i] = eval(defaults['pdf_py'])
    Q_true /= results.h*sum(Q_true)

    # Plot density estimate

    # Make figure and set axis
    plt.figure(figsize=[6, 6])
    ax = plt.subplot(1,1,1)

    # Plot histogram density
    left_bin_edges = results.bin_edges[:-1]
    plt.bar(left_bin_edges, results.R,     width=results.h, linewidth=0, color='gray', zorder=0, alpha=0.5, label='data')

    # Plot deft density estimate
    plt.plot(xs, results.Q_star,     color='blue', linewidth=2, alpha=1, zorder=2, label='deft')

    if input_type == 'simulated':
    # Plot the true density
        plt.plot(xs, Q_true, color='black', linewidth=2, label='true')

    # Tidy up the plot
    #plt.yticks([])
    plt.ylim([0, 2*max(results.Q_star)])
    plt.xlim(results.bbox)
    t = results.deft_1d_compute_time
    plt.title("%s, $\\alpha = %d$, t=%1.2f sec"%(data_type, alpha, t),     fontsize=10)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('density')


    '''
    # deal with evidence ratio later
    # Plot log evidence ratio against t values
    log_Es, ts = results.map_curve.get_log_evidence_ratios()
    plt.plot(ts, log_Es)
    plt.xlabel('t')
    plt.ylabel('log evidence')
    
    '''

    deftFigFile = io.BytesIO()
    plt.savefig(deftFigFile, format='png')
    deftFigFile.seek(0)

    # the following contains the actual data passed to the html template
    deftFigData = base64.b64encode(deftFigFile.getvalue()).decode()

    # Save plot
    #plt.savefig('static/report_test_deft_1d.png')

    return render_template('index.html',result=deftFigData, N=N,G=G,alpha=alpha,data_type=data_type,
                           distribution=sample_distribution_type, dist_index=simulate_index, bbox_left=bbox_left,
                           bbox_right=bbox_right,input_type=input_type,example_data=example_data,
                           example_data_index=example_data_index)


if __name__ == "__main__":
    app.run(debug=True)
    #app.run()
