from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename

#from io import BytesIO
import io
import base64

import uuid
import shutil

from deft_code import deft_1d
from sim import simulate_data_1d

import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
sys.path.append('deft_code/')
TINY_FLOAT64 = sp.finfo(sp.float64).tiny
import os

exec(open(os.getcwd()+"/test_suite/test_header.py").read())

# name of the flask app
app = Flask(__name__)

# this key decrypts the cookie on the client's browser
app.secret_key = os.urandom(32)

# allowed input file extensions
ALLOWED_EXTENSIONS = set(['txt','dat','input'])

# handler methods for checking file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




# deft home page
@app.route('/', methods=['GET','POST'])
def index():

    # dictionary to store all metadata in key value pairs
    metaData = {}

    # create temp files like metadata
    if 'uid' in session:
        print("Temp file already created")
        tempFile = str(session['uid']) + ".input"
        metaDataFile = str(session['uid']) + ".meta"
    else:
        session['uid'] = uuid.uuid4()
        tempFile = str(session['uid']) + ".input"
        metaDataFile = str(session['uid']) + ".meta"
        print("New temp file created")

    sample_distribution_type = ["gaussian", "narrow", "wide", "foothills", "accordian", "goalposts", "towers","uniform",
                                "beta_convex", "beta_concave", "exponential", "gamma", "triangular", "laplace","vonmises"]

    # give the data file the temp name
    dataFileName = tempFile

    # example data list, for keeping drop down updated on re-runs
    example_data = ['buffalo_snowfall.dat','old_faithful_eruption_times.dat','old_faithful_waiting_times.dat','treatment_length.dat']
    # default value
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

    N = 100
    bbox = [-15, 15]
    Z_eval = 'Lap'
    num_Z_samples = 0
    pt_method = 'Lap'
    num_pt_samples = 100
    fix_t_at_t_star = True
    # make this pickalable


    # input_type default value is simulated, user post value will be written to
    # metadata file and update in the html from there.
    input_type = 'simulated'

    # simulate_index that selects default value of distribution list
    simulate_index = 0
    # similar index but for example data
    example_data_index = 0

    # dict of example data
    example_data_dict = {}
    example_data_dict['buffalo_snowfall.dat'] = np.loadtxt('./data/buffalo_snowfall.dat').astype(np.float64)
    example_data_dict['old_faithful_eruption_times.dat'] = np.loadtxt('./data/old_faithful_eruption_times.dat').astype(np.float)
    example_data_dict['old_faithful_waiting_times.dat'] = np.loadtxt('./data/old_faithful_waiting_times.dat').astype(np.float)
    example_data_dict['treatment_length.dat'] = np.loadtxt('./data/treatment_length.dat').astype(np.float)

    # handle posts from web page
    if request.method == 'POST':

        # if run deft button is pressed
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

        # if user data is uploaded
        elif str(request.form.get("dataUploadButton")) == 'Upload Data':
            print("Hitting Upload button ")

            # get file name
            f = request.files['file']
            # secure filename cleans the name of the uploaded file
            f.save(secure_filename(f.filename))

            # write file name in metadata
            metaData['fileName'] = f.filename

            # change radio state to 'User Data'
            input_type = 'user'
            metaData['input_type'] = input_type

            # write all metadata to file
            with open(metaDataFile, "w") as myfile:
                for key in sorted(metaData):
                    myfile.write(key + ":" + str("".join(metaData[key])) + "\n")

            # the following puts uploaded data in the temp file
            # write from
            with open(f.filename) as f1:
                # write to
                with open(dataFileName, "w") as f2:
                    for line in f1:
                        f2.write(line)

    # default data on homepage
    data, defaults = simulate_data_1d.run(data_type, N)

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

    # update radio button value from metadata file
    if 'input_type' in read_metadata_dict:
        input_type = str(read_metadata_dict['input_type']).strip()

    if input_type == 'simulated':
        # Do density estimation
        bbox_left = -15
        bbox_right = 15
        bbox = [bbox_left, bbox_right]
        results = TestCase(N=N, data_seed=0, deft_seed=0, G=100, alpha=3, bbox=bbox, Z_eval=Z_eval,
                           num_Z_samples=num_Z_samples, DT_MAX=1.0, pt_method=pt_method, num_pt_samples=num_pt_samples,
                           fix_t_at_t_star=fix_t_at_t_star).run()

    elif input_type == 'example':
        loaded_data = example_data_dict[example_data_selected]
        bbox_left = int(np.min(loaded_data)-10)
        bbox_right = int(np.max(loaded_data)+10)
        bbox = [bbox_left,bbox_right]
        results = deft_1d.run(loaded_data, G=G, alpha=alpha, bbox=bbox, periodic=False, num_samples=0, print_t=False, tollerance=1E-3)

    elif input_type == 'user' or input_type == 'User Data':

        #user_uploaded_data = np.loadtxt(dataFileName).astype(np.float)
        #bbox_left = int(np.min(user_uploaded_data) - 2)
        #bbox_right = int(np.max(user_uploaded_data) + 2)
        #bbox = [bbox_left, bbox_right]
        #fed_data = np.loadtxt('./data/buffalo_snowfall.dat').astype(np.float64)
        fed_data = example_data_dict['old_faithful_eruption_times.dat'] = np.loadtxt('./data/old_faithful_eruption_times.dat').astype(np.float)
        bbox_left = int(np.min(fed_data)-10)
        bbox_right = int(np.max(fed_data)+10)
        bbox = [bbox_left, bbox_right]
        #results = deft_1d.run(user_uploaded_data, G=G, alpha=alpha, bbox=bbox, periodic=False, num_samples=0, print_t=False,tollerance=1E-3)
        #np.loadtxt('./data/buffalo_snowfall.dat').astype(np.float64)

        results = TestCase(N=N, data_seed=0, deft_seed=0, G=100, alpha=3, bbox=bbox, Z_eval=Z_eval,feed_data=True,data_fed=fed_data,
                           num_Z_samples=num_Z_samples, DT_MAX=1.0, pt_method=pt_method, num_pt_samples=num_pt_samples,
                           fix_t_at_t_star=fix_t_at_t_star).run()
        #print('Input type set to User Data',dataFileName, user_uploaded_data)

    xs = results.results.bin_centers
    phi_samples = results.results.phi_samples
    phi_star = results.results.phi_star

    sample_weights = results.results.phi_weights
    indices = range(num_pt_samples)
    index_probs = sample_weights / sum(sample_weights)
    weighted_sample_indices = np.random.choice(indices, size=num_pt_samples, p=index_probs)
    phi_samples_weighted = phi_samples[:, weighted_sample_indices]

    # Naive Laplace sampling
    xs = results.results.bin_centers
    R = results.results.R
    h = results.results.h
    #Q_true = results.Q_true
    Q_star = results.results.Q_star
    Q_samples = results.results.Q_samples
    sample_weights = results.results.phi_weights

    plt.figure(figsize=[6, 6])
    #plt.figure(1)
    plt.bar(xs, R, width=h, color='grey', alpha=0.3, zorder=2)
    #plt.plot(xs, Q_true, color='black', zorder=3)
    plt.plot(xs, Q_star, color='red', zorder=4)
    plt.plot(xs, Q_samples, color='blue', alpha=0.3, zorder=1)
    #plt.ylim(0, 0.4)
    #plt.show()

    deftFigFile = io.BytesIO()
    plt.savefig(deftFigFile, format='png')
    deftFigFile.seek(0)

    # the following contains the actual data passed to the html template
    deftFigData = base64.b64encode(deftFigFile.getvalue()).decode()

    # Save plot
    #plt.savefig('static/report_test_deft_1d.png')

    '''
    # put in mpl calls here
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
    
    
    deftFigFile = io.BytesIO()
    plt.savefig(deftFigFile, format='png')
    deftFigFile.seek(0)

    # the following contains the actual data passed to the html template
    deftFigData = base64.b64encode(deftFigFile.getvalue()).decode()

    # Save plot
    #plt.savefig('static/report_test_deft_1d.png')
    '''

    #return "<h1>New code working </h1>"

    return render_template('index.html',result=deftFigData, N=N,G=G,alpha=alpha,data_type=data_type,
                           distribution=sample_distribution_type, dist_index=simulate_index, bbox_left=bbox_left,
                           bbox_right=bbox_right,input_type=input_type,example_data=example_data,
                           example_data_index=example_data_index)




if __name__ == "__main__":
    app.run(debug=True)
    #app.run()
