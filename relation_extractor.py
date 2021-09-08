import os
import subprocess
import glob
# import pandas as pd


def stanford_relation_extractor():

    print('Relation Extraction Started')

    for f in glob.glob(os.getcwd() + "/data/output/kg/*.txt"):        
        print("Extracting relations for " + f.split("/")[-1])
        current_directory = os.getcwd()
        print('Current_directory: ' + current_directory)
        os.chdir(current_directory + '/libs/stanford-openie')

        # Check current working directory.
        directory = os.getcwd()
        print("Current working directory %s" % directory)
        print('f: '+f)

        """
        subprocess.Popen(args):
            Execute a child program in a new process.
            
            An example of passing some arguments to an external program as a sequence is:
            Popen(["/usr/bin/git", "commit", "-m", "Fixes a bug."])
            
            or,
            
            args = ['/bin/vikings', '-input', 'eggs.txt', '-output', 'spam spam.txt', '-cmd', "echo '$MONEY'"]
            subprocess.Popen(args) # Success!
        
        subprocess.PIPE: 
            Special value that can be used as the stdin, stdout or stderr argument to Popen 
            and indicates that a pipe to the standard stream should be opened. Most useful with Popen.communicate()
            
        Popen.communicate(input=None, timeout=None)
            Interact with process: Send data to stdin. Read data from stdout and stderr, until end-of-file is reached. 
            Wait for process to terminate and set the returncode attribute. The optional input argument should be data 
            to be sent to the child process, or None, if no data should be sent to the child. 
            If streams were opened in text mode, input must be a string. Otherwise, it must be bytes.

            communicate() returns a tuple (stdout_data, stderr_data). The data will be strings if streams were opened 
            in text mode; otherwise, bytes.

            Note that if you want to send data to the process’s stdin, you need to create the Popen object with 
            stdin=PIPE. Similarly, to get anything other than None in the result tuple, you need to give 
            stdout=PIPE and/or stderr=PIPE too.

            If the process does not terminate after timeout seconds, a TimeoutExpired exception will be raised. 
            Catching this exception and retrying communication will not lose any output.

            The child process is not killed if the timeout expires, so in order to cleanup properly a well-behaved 
            application should kill the child process and finish communication:
        """
        proc = subprocess.Popen(['./process_large_corpus.sh', f, f + '-out.csv'], stdout=subprocess.PIPE)
        output, err = proc.communicate()

    print('Relation Extraction Completed')


if __name__ == '__main__':
    stanford_relation_extractor()
