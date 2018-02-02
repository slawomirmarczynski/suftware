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

    # this list makes the drop down for simulated data
    simulated_distribution_type = ["wide", "narrow", "gaussian", "foothills","random_GM", "accordian", "goalposts", "towers", "towers2","uniform",
                                "beta_convex", "beta_concave", "exponential", "gamma", "triangular", "laplace", "vonmises", "power_law"]

    # posterior sampling method choices
    pt_method_list = ['None','Lap','Lap+W',"MMC"]

    # give the data file the temp name
    dataFileName = tempFile

    # example data list, for keeping drop down updated on re-runs
    example_data = ['buffalo_snowfall.dat','old_faithful_eruption_times.dat','old_faithful_waiting_times.dat','treatment_length.dat']
    # default value
    example_data_selected = 'buffalo_snowfall.dat'

    # write following parameters to metadata files...why?
    # default params used if no post is made
    N = 100
    data_type = 'wide'
    # Deft parameter settings
    G = 100
    alpha = 3
    bbox_left = -10
    bbox_right = 10

    # parameters after sampling code
    #Z_eval = 'Lap'
    Z_eval = 'Lap'
    num_Z_samples = 0
    num_pt_samples = 100
    fix_t_at_t_star = True

    # input_type default value is simulated, user post value will be written to
    # metadata file and update in the html from there.
    input_type = 'simulated'

    # value to maintain the state of the bbox (between manual and auto) between posts
    # default value is 1. 1 == auto, 0 == manual
    bbox_state = 1

    # simulate_index that selects default value of distribution list
    simulate_index = 0
    # similar index but for example data
    example_data_index = 0
    # similar index but for pt_methods
    pt_method_index = 0
    pt_method_selected = 'None' # default value, unless post is made

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
            bbox_left = float(request.form['bbox_left'])
            bbox_right = float(request.form['bbox_right'])
            bbox = [float(request.form['bbox_left']),float(request.form['bbox_right'])]

            bbox_state = len(request.form.getlist('sliderbbox'))
            #print(" Slider Value: ", len(bbox_state))

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
            pt_method_selected = request.form['pt_method']

            # the following two loops and their indices need to be replaced
            # by dicts:

            # loop for selecting the default value of distribution select
            for dist in simulated_distribution_type:
                if str(data_type) == dist:
                    break
                simulate_index += 1

            # loop for selecting the default value of example_data select
            for example in example_data:
                if str(example_data_selected) == example:
                    break
                example_data_index += 1

            # loop for selecting the default value of example_data select
            for method in pt_method_list:
                if pt_method_selected == method:
                    break
                pt_method_index += 1

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

    # select pt_method
    print('index: ',pt_method_index)

    # this because the pt_method cannot take the argument 'None', it has to be None
    if pt_method_index == 0:
        pt_method = None
    else:
        pt_method = pt_method_list[pt_method_index]

    # update radio button value from metadata file
    if 'input_type' in read_metadata_dict:
        input_type = str(read_metadata_dict['input_type']).strip()

    if input_type == 'simulated':

        # Do density estimation
        bbox = [bbox_left, bbox_right]
        results = TestCase(N=N, data_seed=None, deft_seed=None, G=G, alpha=alpha, bbox=bbox, Z_eval=Z_eval,data_type=simulated_distribution_type[simulate_index],
                           num_Z_samples=num_Z_samples, DT_MAX=1.0, pt_method=pt_method, num_pt_samples=num_pt_samples,
                           fix_t_at_t_star=fix_t_at_t_star).run()

    elif input_type == 'example':

        fed_data = example_data_dict[example_data_selected]
        # if bbox state is auto (1), then chose bbox automatically.
        if bbox_state == 1:
            data_spread = np.max(fed_data)-np.min(fed_data)
            bbox_left = int(np.min(fed_data)-0.2*data_spread)
            bbox_right = int(np.max(fed_data)+0.2*data_spread)
            bbox = [bbox_left, bbox_right]

        #results = deft_1d.run(loaded_data, G=G, alpha=alpha, bbox=bbox, periodic=False, num_samples=0, print_t=False, tollerance=1E-3)
        results = TestCase(data_seed=0, deft_seed=0, G=G, alpha=alpha, bbox=bbox, Z_eval=Z_eval, feed_data=True,
                           data_fed=fed_data,
                           num_Z_samples=num_Z_samples, DT_MAX=1.0, pt_method=pt_method, num_pt_samples=num_pt_samples,
                           fix_t_at_t_star=fix_t_at_t_star).run()

    elif input_type == 'user' or input_type == 'User Data':

        user_uploaded_data = np.loadtxt(dataFileName).astype(np.float)

        if bbox_state == 1:
            data_spread = np.max(user_uploaded_data)-np.min(user_uploaded_data)
            bbox_left = int(np.min(user_uploaded_data)-0.2*data_spread)
            bbox_right = int(np.max(user_uploaded_data)+0.2*data_spread)
            bbox = [bbox_left, bbox_right]

        results = TestCase(N=N, data_seed=0, deft_seed=0, G=G, alpha=alpha, bbox=bbox, Z_eval=Z_eval,feed_data=True,data_fed=user_uploaded_data,
                           num_Z_samples=num_Z_samples, DT_MAX=1.0, pt_method=pt_method, num_pt_samples=num_pt_samples,
                           fix_t_at_t_star=fix_t_at_t_star).run()
        #print('Input type set to User Data',dataFileName, user_uploaded_data)

    if pt_method == None:

        xs = results.results.bin_centers
        phi_star = results.results.phi_star

        plt.figure(figsize=[6, 6])
        # plt.figure(1)
        R = results.results.R
        h = results.results.h
        Q_star = results.results.Q_star
        if input_type == 'simulated':
            Q_true = results.Q_true
            plt.plot(xs, Q_true, color='black', zorder=3)
        plt.bar(xs, R, width=h, color='grey', alpha=0.3, zorder=2)
        plt.plot(xs, Q_star, color='red', zorder=4)
        # plt.plot(xs, Q_samples, color='blue', alpha=0.3, zorder=1)

        '''
        print("Q_true")
        for qtrue in Q_true:
            print(qtrue)

        print("Q_star ")
        for qstar in Q_star:
            print(qstar)
        '''

    elif pt_method=='Lap':

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

        if input_type == 'simulated':
            Q_true = results.Q_true
            plt.plot(xs, Q_true, color='black', zorder=3)

        Q_star = results.results.Q_star
        Q_samples = results.results.Q_samples
        sample_weights = results.results.phi_weights

        plt.figure(figsize=[6, 6])
        #plt.figure(1)
        plt.bar(xs, R, width=h, color='grey', alpha=0.3, zorder=2)
        plt.plot(xs, Q_star, color='red', zorder=4)
        plt.plot(xs, Q_samples, color='blue', alpha=0.1, zorder=1)
        #plt.ylim(0, 0.5)

    # importance sampling
    elif pt_method=='Lap+W':

        xs = results.results.bin_centers
        phi_samples = results.results.phi_samples
        phi_star = results.results.phi_star

        sample_weights = results.results.phi_weights
        indices = range(num_pt_samples)
        index_probs = sample_weights / sum(sample_weights)
        weighted_sample_indices = np.random.choice(indices, size=num_pt_samples, p=index_probs)
        phi_samples_weighted = phi_samples[:, weighted_sample_indices]

        R = results.results.R
        h = results.results.h
        if input_type == 'simulated':
            Q_true = results.Q_true
            plt.plot(xs, Q_true, color='black', zorder=3)
            H_true = -sp.sum(Q_true * sp.log2(Q_true + TINY_FLOAT64)) * h

        Q_star = results.results.Q_star
        Q_samples = results.results.Q_samples
        Q_samples_weighted = Q_samples[:, weighted_sample_indices]

        plt.figure(figsize=[6, 6])
        #plt.figure(figsize=[8, 3])
        plt.bar(xs, R, width=h, color='grey', alpha=0.1, zorder=2)
        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        plt.plot(xs, Q_star, color='red', zorder=4)
        #plt.plot(xs, Q_true, color='black', zorder=3,label="$Q_{True}$")
        #plt.plot(xs, Q_star, color='red', zorder=4, label="$Q^*$")
        plt.plot(xs, Q_samples_weighted[:,0:100], color='blue', alpha=0.1, zorder=1)
        #plt.ylim(0, 0.5)


        # Compute entropy of Q_true and Q_samples

        H_samples = np.zeros(num_pt_samples)
        for k in range(num_pt_samples):
            Q_k = Q_samples[:, k]
            H_samples[k] = -sp.sum(Q_k * sp.log2(Q_k + TINY_FLOAT64)) * h
        # Naive bias & spread
        #H_mean_naive = sp.mean(H_samples)
        #H_std_naive = sp.std(H_samples)
        #H_bias[i, 0] = H_mean_naive - H_true
        #H_spread[i, 0] = H_std_naive
        # Weighted bias & spread
        H_mean_weighted = sp.sum(H_samples * sample_weights) / sp.sum(sample_weights)
        H_std_weighted = sp.sqrt(
            sp.sum(H_samples ** 2 * sample_weights) / sp.sum(sample_weights) - H_mean_weighted ** 2)

        #H_bias[i, 1] = H_mean_weighted - H_true
        #H_spread[i, 1] = H_std_weighted
        plt.tick_params(axis='both', which='major', labelsize=16)
        if input_type=='simulated':
            plt.title("$H_{True}$=%1.2f bits, $H_{estimated}$=%1.2f $\pm$ %1.2f bits" % (H_true,H_mean_weighted,H_std_weighted), fontsize=18)
        else:
            plt.title("$H_{estimated}$=%1.2f $\pm$ %1.2f bits" % (H_mean_weighted, H_std_weighted), fontsize=18)
        plt.legend(fontsize=18)
        plt.xlabel('x',fontsize=18)
        plt.ylabel('Density',fontsize=18)
        #plt.ylim(0, 0.3)
        #plt.figure(num=None,figsize=(8,2),dpi=80)
        plt.tight_layout()
        #plt.savefig("test.pdf")

    elif pt_method=='MMC':

        xs = results.results.bin_centers
        phi_samples = results.results.phi_samples
        phi_star = results.results.phi_star

        # MMC sampling
        xs = results.results.bin_centers
        R = results.results.R
        h = results.results.h
        if input_type=='simulated':
            Q_true = results.Q_true
            plt.plot(xs, Q_true, color='black', zorder=3)
        Q_star = results.results.Q_star
        Q_samples = results.results.Q_samples

        plt.figure(figsize=[6, 6])
        #plt.figure(1)
        plt.bar(xs, R, width=h, color='grey', alpha=0.3, zorder=2)
        #
        plt.plot(xs, Q_star, color='red', zorder=4)
        plt.plot(xs, Q_samples, color='blue', alpha=0.3, zorder=1)
        #plt.ylim(0, 0.5)

    deftFigFile = io.BytesIO()
    plt.savefig(deftFigFile, format='png')
    deftFigFile.seek(0)

    # the following contains the actual data passed to the html template
    deftFigData = base64.b64encode(deftFigFile.getvalue()).decode()

    # Save plot
    plt.savefig('static/report_test_deft_1d.png')

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

    return render_template('index.html',result=deftFigData, N=N,G=G,alpha=alpha,data_type=data_type,
                           distribution=simulated_distribution_type, dist_index=simulate_index, bbox_left=bbox_left,
                           bbox_right=bbox_right,input_type=input_type,example_data=example_data,
                           example_data_index=example_data_index,pt_method_list=pt_method_list,pt_method_index=pt_method_index,
                           bbox_state=bbox_state)




if __name__ == "__main__":
    app.run(debug=True)
    #app.run()
