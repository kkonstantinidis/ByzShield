from __future__ import print_function
import sys
import threading
import Queue
import paramiko as pm
import boto3
import time
import json
import os


class Cfg(dict):

   def __getitem__(self, item):
       item = dict.__getitem__(self, item)
       if type(item) == type([]): # ~ the value is a list
           return [x % self if type(x) == type("") else x for x in item] # ~ return string or int for each element, see below
       if type(item) == type(""): # ~ the value is a string
           return item % self # ~ if string has a specifier for a previous dictionary key it substitutes it
       return item # ~ the value is a number

cfg = Cfg({
    "name" : "Timeout",      # Unique name for this specific configuration
    "key_name": "virginiakey",          # ~ Necessary to ssh into created instances, WITHOUT .pem
    # Cluster topology
    "n_masters" : 1,                      # Should always be 1
    "n_workers" : 1,
    "num_replicas_to_aggregate" : "8", # deprecated, not necessary
    "method" : "spot",
    # Region speficiation
    "region" : "us-east-1",
    "availability_zone" : "us-east-1d",
    # Machine type - instance type configuration.
    "master_type" : "i3.16xlarge",
    "worker_type" : "r3.large",
    # please only use this AMI for pytorch
    "image_id": "ami-017571e50bb10c6b4",
    # Launch specifications
    "spot_price" : "10",                 # Has to be a string
    # SSH configuration
    "ssh_username" : "ubuntu",            # For sshing. E.G: ssh ssh_username@hostname
    "path_to_keyfile" : "virginiakey.pem", # ~ be careful with this path since the execution path depends on where you run the code from

    # NFS configuration
    # To set up these values, go to Services > Elastic File System > Create file system, and follow the directions.
    "nfs_ip_address" : "172.31.18.129",          # us-east-1c
    "nfs_mount_point" : "/home/ubuntu/shared",       # NFS base dir
    "base_out_dir" : "%(nfs_mount_point)s/%(name)s", # Master writes checkpoints to this directory. Outfiles are written to this directory.
    "setup_commands" :
    [
        "mkdir %(base_out_dir)s",
    ],
    # Command specification
    # Master pre commands are run only by the master
    "master_pre_commands" :
    [
        "cd my_mxnet",
        "git fetch && git reset --hard origin/master",
        "cd cifar10",
        "ls",
    ],
    # Pre commands are run on every machine before the actual training.
    "pre_commands" :
    [
        "cd my_mxnet",
        "git fetch && git reset --hard origin/master",
        "cd cifar10",
    ],
    # Model configuration
    "batch_size" : "4", # ~ never used
    "max_steps" : "2000", # ~ never used
    "initial_learning_rate" : ".001", # ~ never used
    "learning_rate_decay_factor" : ".9", # ~ never used
    "num_epochs_per_decay" : "1.0", # ~ never used
    # Train command specifies how the ps/workers execute tensorflow.
    # PS_HOSTS - special string replaced with actual list of ps hosts.
    # TASK_ID - special string replaced with actual task index.
    # JOB_NAME - special string replaced with actual job name.
    # WORKER_HOSTS - special string replaced with actual list of worker hosts
    # ROLE_ID - special string replaced with machine's identity (E.G: master, worker0, worker1, ps, etc)
    # %(...)s - Inserts self referential string value.
    "train_commands" :
    [
        "echo ========= Start ==========="
    ],
    "security_group": ["mpi_security"],
})

def mxnet_ec2_run(argv, configuration):
    client = boto3.client("ec2", region_name=configuration["region"]) # ~ create a low-level service client by name
    ec2 = boto3.resource("ec2", region_name=configuration["region"]) # ~ create a resource service client by name

    def sleep_a_bit():
        time.sleep(5)

    # ~ prints how many machines of each type are running
    # instances: TBD
    def summarize_instances(instances):
        instance_type_to_instance_map = {}
        for instance in sorted(instances, key=lambda x:x.id):
            typ = instance.instance_type
            if typ not in instance_type_to_instance_map:
                instance_type_to_instance_map[typ] = []
            instance_type_to_instance_map[typ].append(instance)

        for type in instance_type_to_instance_map:
            print("Type\t", type)
            for instance in instance_type_to_instance_map[type]:
                print("instance\t", instance, "\t", instance.public_ip_address)
            print

        for k,v in instance_type_to_instance_map.items(): # ~ for each key-value pair in dictionary
            print("%s - %d running" % (k, len(v)))

        return instance_type_to_instance_map

    # ~ prints how many machines of each type are running BUT not running tensorflow
    def summarize_idle_instances(argv):
        print("Idle instances: (Idle = not running tensorflow)")
        summarize_instances(get_idle_instances())

    # ~ boto3 query to EC2 to return running instances
    def summarize_running_instances(argv):
        print("Running instances: ")
        # ~ instance-state-name: the state of the instance (pending | running | shutting-down | terminated | stopping | stopped ), so here we only filter the running ones
        # SOS
        # filter() creates an iterable of all Instance resources in the collection filtered by kwargs passed to method and it returns list(ec2.Instance) so all attributes of the instances are available
        summarize_instances(ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}, {'Name': 'key-name', 'Values': [configuration["key_name"]]}]))

    # Terminate all request.
    def terminate_all_requests():
         spot_requests = client.describe_spot_instance_requests()
         spot_request_ids = []
         # ~ "SpotInstanceRequests": main key of returned dictionary, value is a list of requests (dicts)
         for spot_request in spot_requests["SpotInstanceRequests"]:
            # ~ "State": key of request dictionary
            if spot_request["State"] != "cancelled" and spot_request["LaunchSpecification"]["KeyName"] == configuration["key_name"]: # not yet cancelled
               spot_request_id = spot_request["SpotInstanceRequestId"]
               spot_request_ids.append(spot_request_id) # ~ to be terminated

         if len(spot_request_ids) != 0:
             print("Terminating spot requests: %s" % " ".join([str(x) for x in spot_request_ids]))
             client.cancel_spot_instance_requests(SpotInstanceRequestIds=spot_request_ids)

         # Wait until all are cancelled.
         done = False
         while not done:
             print("Waiting for all spot requests to be terminated...")
             done = True
             spot_requests = client.describe_spot_instance_requests()
             states = [x["State"] for x in spot_requests["SpotInstanceRequests"] if x["LaunchSpecification"]["KeyName"] == configuration["key_name"]]
             for state in states:
                 if state != "cancelled":
                     done = False
             sleep_a_bit()

    # Terminate all instances in the configuration
    # Note: all_instances = ec2.instances.all() to get all intances
    def terminate_all_instances():
        live_instances = ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}, {'Name': 'key-name', 'Values': [configuration["key_name"]]}])
        all_instance_ids = [x.id for x in live_instances]
        print([x.id for x in live_instances])
        if len(all_instance_ids) != 0:
            print("Terminating instances: %s" % (" ".join([str(x) for x in all_instance_ids])))
            client.terminate_instances(InstanceIds=all_instance_ids) # ~ terminates the instance IDs in the list

            # Wait until all are terminated
            done = False
            while not done:
                print("Waiting for all instances to be terminated...")
                done = True
                instances = ec2.instances.all()
                for instance in instances:
                    if instance.state == "active":
                        done = False
                sleep_a_bit()

    # Launch instances as specified in the configuration.
    def launch_instances():
        method = configuration["method"]
        worker_instance_type, worker_count = configuration["worker_type"], configuration["n_workers"]
        master_instance_type, master_count = configuration["master_type"], configuration["n_masters"]
        specs = [(worker_instance_type, worker_count), (master_instance_type, master_count)]
        for (instance_type, count) in specs:
            launch_specs = {"KeyName" : configuration["key_name"],
            "ImageId" : configuration["image_id"],
            "InstanceType" : instance_type,
            "Placement" : {"AvailabilityZone":configuration["availability_zone"]},
            "SecurityGroups": configuration["security_group"]}
            if method == "spot":
                client.request_spot_instances(InstanceCount=count, LaunchSpecification=launch_specs, SpotPrice=configuration["spot_price"])
            elif method == "reserved":
                client.run_instances(ImageId=launch_specs["ImageId"],
                MinCount=count,
                MaxCount=count,
                KeyName=launch_specs["KeyName"],
                InstanceType=launch_specs["InstanceType"],
                Placement=launch_specs["Placement"],
                SecurityGroups=launch_specs["SecurityGroups"])
            else:
                print("Unknown method: %s" % method)
                sys.exit(-1)

    def wait_until_running_instances_initialized():
        done = False
        while not done:
            print("Waiting for instances to be initialized...")
            done = True
            live_instances = ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}, {'Name': 'key-name', 'Values': [configuration["key_name"]]}])
            ids = [x.id for x in live_instances]
            
            # ~ describes the instance IDs in the list
            resps_list = [client.describe_instance_status(InstanceIds=ids[i:i+50]) for i in range(0, len(ids), 50)] # +50 ???
            
            statuses = []
            for resp in resps_list:
                # ~ "InstanceStatuses": main key of returned dictionary, value is a list of statuses (dicts)
                statuses += [x["InstanceStatus"]["Status"] for x in resp["InstanceStatuses"]]
            print(statuses)
            done = statuses.count("ok") == len(statuses) # ~ "ok" status is initialized, checks whether all running statuses are ok
            if len(ids) <= 0:
                done = False
            sleep_a_bit()

    # Waits until status requests are all fulfilled.
    # Prints out status of request in between time waits.
    def wait_until_instance_request_status_fulfilled():
         requests_fulfilled = False
         n_active_or_open = 0
         while not requests_fulfilled or n_active_or_open == 0:
             requests_fulfilled = True
             statuses = client.describe_spot_instance_requests()
             print("InstanceRequestId, InstanceType, SpotPrice, State - Status : StatusMessage")
             print("-------------------------------------------")
             n_active_or_open = 0
             for instance_request in statuses["SpotInstanceRequests"]: # ~ see function terminate_all_requests()
                 if instance_request["LaunchSpecification"]["KeyName"] != configuration["key_name"]:
                    continue
                 sid = instance_request["SpotInstanceRequestId"]
                 machine_type = instance_request["LaunchSpecification"]["InstanceType"]
                 price = instance_request["SpotPrice"]
                 state = instance_request["State"]
                 status, status_string = instance_request["Status"]["Code"], instance_request["Status"]["Message"] # ~ status code & description for the status code
                 if state == "active" or state == "open":
                     n_active_or_open += 1
                     print("%s, %s, %s, %s - %s : %s" % (sid, machine_type, price, state, status, status_string))
                     if state != "active":
                         requests_fulfilled = False
             print("-------------------------------------------")
             sleep_a_bit()

    # ~ Create a SSH client an instance
    # instance: boto3.ec2.Instance
    def connect_client(instance):
        client = pm.SSHClient()
        host = instance.public_ip_address
        
        # ~ If the server's hostname is not found, the missing host key policy is used, here this is automatically adding the hostname and new host key to the local HostKeys object, and saving it
        client.set_missing_host_key_policy(pm.AutoAddPolicy())
        
        # test
        # print("X", configuration["ssh_username"], configuration["path_to_keyfile"])
        
        client.connect(host, username=configuration["ssh_username"], key_filename=configuration["path_to_keyfile"])
        return client

    # Takes a list of commands (E.G: ["ls", "cd models"]
    # and executes command on instance, returning the stdout.
    # Executes everything in one session, and returns all output from all the commands.
    # ~ instance: boto3.ec2.Instance
    def run_ssh_commands(instance, commands):
        done = False
        while not done:
           try:
              print("Instance %s, Running ssh commands:\n%s\n" % (instance.public_ip_address, "\n".join(commands)))

              # ~ Always need to exit SSH session
              commands.append("exit")

              # Set up ssh client
              client = connect_client(instance)

              # Clear the stdout from ssh'ing in
              # For each command perform command and read stdout
              commandstring = "\n".join(commands) # ~ different commands are separated by newline so that all can be executed (paramiko)
              stdin, stdout, stderr = client.exec_command(commandstring) # ~ stdin, stdout, and stderr are returned as Python file-like objects (paramiko)
              output = stdout.read()

              # Close down
              stdout.close()
              stdin.close()
              client.close()
              done = True
           except Exception as e:
              done = False
              print(e.message)
        return output

    # ~ instance: boto3.ec2.Instance
    def run_ssh_commands_parallel(instance, commands, q):
        output = run_ssh_commands(instance, commands)
        q.put((instance, output)) # ~ put item into the queue, block if necessary until a free slot is available

    # Checks whether instance is idle. Assumed that instance is up and running.
    # An instance is idle if it is not running tensorflow...
    # Returns a tuple of (instance, is_instance_idle). We return a tuple for multithreading ease.
    # ~ instance: boto3.ec2.Instance
    def is_instance_idle(q, instance):
        python_processes = run_ssh_commands(instance, ["ps aux | grep python"]) # ~ get all running Python processes
        q.put((instance, not "ps_hosts" in python_processes and not "ps_workers" in python_processes)) # ~ ps_hosts & ps_workers are tensorflow or AWS keywords for a Python process ???

    # Idle instances are running instances that are not running the inception model.
    # We check whether an instance is running the inception model by ssh'ing into a running machine,
    # and checking whether python is running.
    def get_idle_instances():
        live_instances = ec2.instances.filter(
            Filters=[{'Name': 'instance-state-name', 'Values': ['running']},
                     {'Name': 'key-name', 'Values': [configuration["key_name"]]}])
        threads = []
        q = Queue.Queue() # ~ FIFO queue

        # Run commands in parallel, writing to the queue
        for instance in live_instances:
            # ~ target: the callable object to be invoked by the thread's run() method
            # args: the argument tuple for the target invocation, its elements (here only 2) will be passed to this function with thread's run()
            t = threading.Thread(target=is_instance_idle, args=(q, instance))
            t.daemon = True # ~ this thread is a daemon thread 
            t.start() # ~ start the thread's activity, will call thread's run()
            threads.append(t)

        # Wait for threads to finish
        for thread in threads:
            thread.join()

        # Collect idle instances
        idle_instances = []
        while not q.empty():
            instance, is_idle = q.get() # ~ remove and return an item from the queue
            if is_idle:
                idle_instances.append(instance)

        return idle_instances

    # ~ get the requirements (instance type & counts for PS and workers) from configuration
    def get_instance_requirements():
        worker_instance_type, worker_count = configuration["worker_type"], configuration["n_workers"]
        master_instance_type, master_count = configuration["master_type"], configuration["n_masters"]
        specs = [(worker_instance_type, worker_count),
                 (master_instance_type, master_count)]
        reqs = {}
        for (type_needed, count_needed) in specs:
            if type_needed not in reqs:
                reqs[type_needed] = 0
            reqs[type_needed] += count_needed
        return reqs

    # Returns whether the idle instances satisfy the specs of the configuration.
    def check_idle_instances_satisfy_configuration():
        # Create a map of instance types to instances of that type
        idle_instances = get_idle_instances()
        instance_type_to_instance_map = summarize_instances(idle_instances)

        # Get instance requirements
        reqs = get_instance_requirements()

        # Check the requirements are satisfied.
        print("Checking whether # of running instances satisfies the configuration...")
        for k,v in instance_type_to_instance_map.items():
            n_required = 0 if k not in reqs else reqs[k]
            print("%s - %d running vs %d required" % (k,len(v),n_required))
            
            # ~ checks if the list of idle boto3.ec2.Instance for type "k" is less than required count
            if len(v) < n_required:
                print("Error, running instances failed to satisfy configuration requirements")
                sys.exit(0)
        print("Success, running instances satisfy configuration requirement")

    def shut_everything_down(argv):
        terminate_all_requests()
        terminate_all_instances()

    # ~ never used, see help_mapS
    def run_mxnet_grid_search(argv, port=1334):
        # ~ Check idle instances satisfy configuration count requirements
        check_idle_instances_satisfy_configuration()

        # ~ Get idle instances (list of boto3.ec2.Instance)
        idle_instances = get_idle_instances()

        # Assign instances for worker/ps/etc
        instance_type_to_instance_map = summarize_instances(idle_instances)
        specs = {
            "master" : {"instance_type" : configuration["master_type"],
                        "n_required" : configuration["n_masters"]},
            "worker" : {"instance_type" : configuration["worker_type"],
                        "n_required" : configuration["n_workers"]}
        }
        machine_assignments = {
            "master" : [],
            "worker" : []
        }
        for role, requirement in sorted(specs.items(), key=lambda x:x[0]): # ~ sorted() just gets master config before worker config since dictionaries are not ordered iterables
            instance_type_for_role = requirement["instance_type"]
            n_instances_needed = requirement["n_required"]
            
            # ~ needed instances to be assigned and remaining (list of boto3.ec2.Instance)
            instances_to_assign, rest = instance_type_to_instance_map[instance_type_for_role][:n_instances_needed], instance_type_to_instance_map[instance_type_for_role][n_instances_needed:]
            
            # ~ needed instances information for this machine type is stored, discard them from hashtable and keep only the rest
            instance_type_to_instance_map[instance_type_for_role] = rest
            
            # ~ needed instances for role (master or worker)
            machine_assignments[role] = instances_to_assign

        # Construct the host strings necessary for running the inception command.
        # Note we use private ip addresses to avoid EC2 transfer costs.
        worker_host_string = ",".join([x.private_ip_address+":"+str(port) for x in machine_assignments["master"] + machine_assignments["worker"]])

        # ~ Create a map of command & machine assignments
        command_machine_assignments = {}
        setup_machine_assignments = {} # ~ NOT POPULATED LATER ???

        # ~ Construct the MASTER command
        command_machine_assignments["master"] = {"instance" : machine_assignments["master"][0], "commands" : list(configuration["master_pre_commands"])} # add key to dict
        for command_string in configuration["train_commands"]: # ~ supposed to be a list but it's just an echo command here ???
            command_machine_assignments["master"]["commands"].append(command_string.replace("JOB_NAME", "worker").replace("WORKER_HOSTS", worker_host_string).replace("ROLE_ID", "master")) # just some string replacements
        print(command_machine_assignments)

        # ~ Construct the WORKER commands
        for worker_id, instance in enumerate(machine_assignments["worker"]):
            name = "worker_%d" % worker_id
            # ~ similar to master code
            command_machine_assignments[name] = {"instance" : instance, "commands" : list(configuration["pre_commands"])}
            for command_string in configuration["train_commands"]:
                command_machine_assignments[name]["commands"].append(command_string.replace("TASK_ID", "%d" % (worker_id+1)).replace("JOB_NAME", "worker").replace("WORKER_HOSTS", worker_host_string).replace("ROLE_ID", name))

        print(command_machine_assignments)

        # ~ DOES NOT DO ANYTHING, SEE ABOVE
        # Run the commands via ssh in parallel
        threads = []
        q = Queue.Queue()
        # ~ for master and all workers
        for name, command_and_machine in setup_machine_assignments.items():
            instance = command_and_machine["instance"]
            commands = command_and_machine["commands"]
            print("-----------------------")
            print("Pre Command: %s\n" % " ".join(commands))
            t = threading.Thread(target=run_ssh_commands_parallel, args=(instance, commands, q))
            t.start()
            threads.append(t)
        # Wait until commands are all finished
        for t in threads:
            t.join()

        threads = []
        q = Queue.Queue()

        running_process = 0
        for name, command_and_machine in command_machine_assignments.items():
            instance = command_and_machine["instance"] # ~ get the boto3.ec2.Instance for this element
            
            # ~ WHERE IS THIS FILE train_cifar10.py ???
            neo_commands = "python train_cifar10.py --running_mode=grid_search --gpus=0 "\
                           "--running_process={} "\
                           "--batch-size={} "\
                           "--dir={}/grid_search> {}/grid_search/batch_size_{}/running_{}_process.out 2>&1 &".format(
                running_process,
                configuration['batch_size'],
                configuration['nfs_mount_point'],
                configuration['nfs_mount_point'],
                configuration['batch_size'],
                running_process)

            commands = command_and_machine["commands"]
            commands.append('mkdir {}/grid_search'.format(configuration['nfs_mount_point']))
            commands.append('mkdir {}/grid_search/batch_size_{}'.format(
                configuration['nfs_mount_point'],
                configuration['batch_size']))
            commands.append(neo_commands)

            print("-----------------------")
            print("Command: %s\n" % " ".join(commands))
            t = threading.Thread(target=run_ssh_commands_parallel, args=(instance, commands, q))
            t.start()
            threads.append(t)
            running_process += 1

        # Wait until commands are all finished
        for t in threads:
            t.join()

        # Print the output
        while not q.empty():
            instance, output = q.get()
            print(instance.public_ip_address)
            print(output)

        # Debug print
        instances = []
        print("\n--------------------------------------------------\n")
        print("Machine assignments:")
        print("------------------------")
        for name, command_and_machine in command_machine_assignments.items():
            instance = command_and_machine["instance"]
            instances.append(instance)
            commands = command_and_machine["commands"]
            ssh_command = "ssh -i %s %s@%s" % (configuration["path_to_keyfile"], configuration["ssh_username"], instance.public_ip_address) # ~ normal SSH using key
            print("%s - %s" % (name, instance.instance_id))
            print("To ssh: %s" % ssh_command)
            print("------------------------")

        # Print out list of instance ids (which will be useful in selctively stopping inception
        # for given instances.
        instance_cluster_string = ",".join([x.instance_id for x in instances])
        print("\nInstances cluster string: %s" % instance_cluster_string)

        # Print out the id of the configuration file
        cluster_save = {
            "configuration" : configuration,
            "name" : configuration["name"],
            "command_machine_assignments" : command_machine_assignments,
            "cluster_string" : instance_cluster_string
        }

        return cluster_save

    # ~ similar to previous one
    # never used, SEE help_map
    def run_mxnet_loss_curve(argv, port=1334):
        # Check idle instances satisfy configs
        check_idle_instances_satisfy_configuration()

        # Get idle instances
        idle_instances = get_idle_instances()

        # Assign instances for worker/ps/etc
        instance_type_to_instance_map = summarize_instances(idle_instances)
        specs = {
            "master" : {"instance_type" : configuration["master_type"],
                        "n_required" : configuration["n_masters"]},
            "worker" : {"instance_type" : configuration["worker_type"],
                        "n_required" : configuration["n_workers"]}
        }
        machine_assignments = {
            "master" : [],
            "worker" : []
        }
        for role, requirement in sorted(specs.items(), key=lambda x:x[0]):
            instance_type_for_role = requirement["instance_type"]
            n_instances_needed = requirement["n_required"]
            instances_to_assign, rest = instance_type_to_instance_map[instance_type_for_role][:n_instances_needed], instance_type_to_instance_map[instance_type_for_role][n_instances_needed:]
            instance_type_to_instance_map[instance_type_for_role] = rest
            machine_assignments[role] = instances_to_assign

        # Construct the host strings necessary for running the inception command.
        # Note we use private ip addresses to avoid EC2 transfer costs.
        worker_host_string = ",".join([x.private_ip_address+":"+str(port) for x in machine_assignments["master"] + machine_assignments["worker"]])

        # Create a map of command&machine assignments
        command_machine_assignments = {}
        setup_machine_assignments = {}

        # Construct the master command
        command_machine_assignments["master"] = {"instance" : machine_assignments["master"][0], "commands" : list(configuration["master_pre_commands"])}
        for command_string in configuration["train_commands"]:
            command_machine_assignments["master"]["commands"].append(command_string.replace("JOB_NAME", "worker").replace("WORKER_HOSTS", worker_host_string).replace("ROLE_ID", "master"))
        print(command_machine_assignments)

        # Construct the worker commands
        for worker_id, instance in enumerate(machine_assignments["worker"]):
            name = "worker_%d" % worker_id
            command_machine_assignments[name] = {"instance" : instance,
                                                 "commands" : list(configuration["pre_commands"])}
            for command_string in configuration["train_commands"]:
                command_machine_assignments[name]["commands"].append(command_string.replace("TASK_ID", "%d" % (worker_id+1)).replace("JOB_NAME", "worker").replace("WORKER_HOSTS", worker_host_string).replace("ROLE_ID", name))

        print(command_machine_assignments)

        # Run the commands via ssh in parallel
        threads = []
        q = Queue.Queue()

        for name, command_and_machine in setup_machine_assignments.items():
            instance = command_and_machine["instance"]
            commands = command_and_machine["commands"]
            print("-----------------------")
            print("Pre Command: %s\n" % " ".join(commands))
            t = threading.Thread(target=run_ssh_commands_parallel, args=(instance, commands, q))
            t.start()
            threads.append(t)

        # Wait until commands are all finished
        for t in threads:
            t.join()

        threads = []
        q = Queue.Queue()

        batch_size_list = [4, 32, 50, 100, 500, 1000]
        learning_rate_list = [0.046, 0.05, 0.068, 0.068, 0.048, 0.086]
        running_process = 0
        for name, command_and_machine in command_machine_assignments.items():
            instance = command_and_machine["instance"]
            neo_commands = "python train_cifar10.py --running_mode=training --gpus=0 "\
                           "--batch-size={} "\
                           "--lr={} "\
                           "--model-prefix={}/model_checkpoints/batch_size_{} "\
                           "--dir={}/loss_curve > "\
                           "{}/loss_curve/running_batch_size_{}.out 2>&1 &".format(
                batch_size_list[running_process],
                learning_rate_list[running_process],
                configuration['nfs_mount_point'],
                batch_size_list[running_process],
                configuration['nfs_mount_point'],
                configuration['nfs_mount_point'],
                batch_size_list[running_process])

            commands = command_and_machine["commands"]
            commands.append('mkdir {}/model_checkpoints/'.format(configuration['nfs_mount_point']))
            commands.append('mkdir {}/loss_curve'.format(configuration['nfs_mount_point']))
            commands.append(neo_commands)

            print("-----------------------")
            print("Command: %s\n" % " ".join(commands))
            t = threading.Thread(target=run_ssh_commands_parallel, args=(instance, commands, q))
            t.start()
            threads.append(t)
            running_process += 1

        # Wait until commands are all finished
        for t in threads:
            t.join()

        # Print the output
        while not q.empty():
            instance, output = q.get()
            print(instance.public_ip_address)
            print(output)

        # Debug print
        instances = []
        print("\n--------------------------------------------------\n")
        print("Machine assignments:")
        print("------------------------")
        for name, command_and_machine in command_machine_assignments.items():
            instance = command_and_machine["instance"]
            instances.append(instance)
            commands = command_and_machine["commands"]
            ssh_command = "ssh -i %s %s@%s" % (configuration["path_to_keyfile"], configuration["ssh_username"], instance.public_ip_address)
            print("%s - %s" % (name, instance.instance_id))
            print("To ssh: %s" % ssh_command)
            print("------------------------")

        # Print out list of instance ids (which will be useful in selctively stopping inception
        # for given instances.
        instance_cluster_string = ",".join([x.instance_id for x in instances])
        print("\nInstances cluster string: %s" % instance_cluster_string)

        # Print out the id of the configuration file
        cluster_save = {
            "configuration" : configuration,
            "name" : configuration["name"],
            "command_machine_assignments" : command_machine_assignments,
            "cluster_string" : instance_cluster_string
        }

        return cluster_save


    # ~ similar to run_mxnet_grid_search(), read comments there
    # saves machine (PS and workers) hostnames in 3 formats: "{IP} \t deeplearning-worker{COUNTER}", alias, private IP 
    def get_hosts(argv, port=22):
        # Check idle instances satisfy configs
        check_idle_instances_satisfy_configuration()

        # Get idle instances
        idle_instances = get_idle_instances()

        # Assign instances for worker/ps/etc
        instance_type_to_instance_map = summarize_instances(idle_instances)
        specs = {
            "master" : {"instance_type" : configuration["master_type"],
                        "n_required" : configuration["n_masters"]},
            "worker" : {"instance_type" : configuration["worker_type"],
                        "n_required" : configuration["n_workers"]}
        }
        machine_assignments = {
            "master" : [],
            "worker" : []
        }
        for role, requirement in sorted(specs.items(), key=lambda x:x[0]):
            instance_type_for_role = requirement["instance_type"]
            n_instances_needed = requirement["n_required"]
            instances_to_assign, rest = instance_type_to_instance_map[instance_type_for_role][:n_instances_needed], instance_type_to_instance_map[instance_type_for_role][n_instances_needed:]
            instance_type_to_instance_map[instance_type_for_role] = rest
            machine_assignments[role] = instances_to_assign

        # Construct the host strings necessary for running the inception command.
        # Note we use private ip addresses to avoid EC2 transfer costs.
        # ~ saves each machine in format {IP}\tdeeplearning-worker{COUNTER}
        worker_host_string = ",".join([x.private_ip_address+":"+str(port) for x in machine_assignments["master"] + machine_assignments["worker"]])
        hosts_out = open('hosts', 'w') # ~ hosts file to store at
        print('master ip ', machine_assignments['master'][0].public_ip_address)
        count = 0
        for instance in machine_assignments["master"] + machine_assignments["worker"]:
            count += 1
            print('{}\tdeeplearning-worker{}'.format(instance.private_ip_address, count), end='\n', file=hosts_out)
        hosts_out.flush() # ~ forces usaved file buffer to disk if it's still in an OS buffer
        hosts_out.close()

        hosts_alias_out = open('hosts_alias', 'w')
        count = 0
        for _ in machine_assignments["master"] + machine_assignments["worker"]:
            count += 1
            print('deeplearning-worker{}'.format(count), end='\n', file=hosts_alias_out)
        hosts_alias_out.flush()
        hosts_alias_out.close()

        hosts_alias_out = open('hosts_address', 'w')
        count = 0
        for instance in machine_assignments["master"] + machine_assignments["worker"]:
            count += 1
            print('{}'.format(instance.private_ip_address), end='\n', file=hosts_alias_out)
        hosts_alias_out.flush()
        hosts_alias_out.close()
        return

    # ~ argv will be [python, kill_python, "instance_id1,instance_id2,..."]
    # inception_ec2.py is useless but need to change expected argv
    def kill_python(argv):
        if len(argv) != 3:
            print("Usage: python inception_ec2.py kill_python instance_id1,instance_id2,id3...") # ~ it should be "Usage: python kill_python instance_id1,instance_id2,id3..." ???
            sys.exit(0)
        cluster_instance_string = argv[2]
        instance_ids_to_shutdown = cluster_instance_string.split(",")

        live_instances = ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}])
        threads = []
        q = Queue.Queue()
        for instance in live_instances:
            if instance.instance_id in instance_ids_to_shutdown:
                commands = ["sudo pkill -9 python"] # ~ kills all Python processes
                t = threading.Thread(target=run_ssh_commands_parallel, args=(instance, commands, q))
                t.start()
                threads.append(t)
        for thread in threads:
            thread.join()
        summarize_idle_instances(None)

    # ~ same as previous but for all instances
    def kill_all_python(argv):
        live_instances = ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']},  {'Name': 'key-name', 'Values': [configuration["key_name"]]}])
        threads = []
        q = Queue.Queue()
        for instance in live_instances:
            commands = ["sudo pkill -9 python"]
            t = threading.Thread(target=run_ssh_commands_parallel, args=(instance, commands, q))
            t.start()
            threads.append(t)
        for thread in threads:
            thread.join()
        summarize_idle_instances(None)

    # ~ argv will be [python, inception_ec2.py, run_command, "instance_id1,instance_id2,..."]
    # inception_ec2.py is useless but need to change expected argv
    # quiet: to print or not
    def run_command(argv, quiet=False):
        if len(argv) != 4:
            print("Usage: python inception_ec2.py run_command instance_id1,instance_id2,id3... command") # ~ it should be "Usage: python run_command instance_id1,instance_id2,id3..." ???
            sys.exit(0)
        cluster_instance_string = argv[2]
        command = argv[3]
        instance_ids_to_run_command = cluster_instance_string.split(",")

        live_instances = ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}, {'Name': 'key-name', 'Values': [configuration["key_name"]]}])
        threads = []
        q = Queue.Queue()
        for instance in live_instances:
            if instance.instance_id in instance_ids_to_run_command:
                commands = [command]
                t = threading.Thread(target=run_ssh_commands_parallel, args=(instance, commands, q))
                t.start()
                threads.append(t)
        for thread in threads:
            thread.join()

        while not q.empty():
            instance, output = q.get()
            if not quiet:
                print(instance, output)

    # Setup nfs on all instances
    # ~ inception_ec2.py is useless but need to change expected argv
    def setup_nfs(argv):
        print("Clearing previous nfs file system...")
        live_instances = ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}, {'Name': 'key-name', 'Values': [configuration["key_name"]]}, {'Name': 'key-name', 'Values': [configuration["key_name"]]}])
        live_instances_string = ",".join([x.instance_id for x in live_instances])
        rm_command = "sudo rm -rf %s" % configuration["nfs_mount_point"]
        argv = ["python", "inception_ec2.py", live_instances_string, rm_command] # ~ inception_ec2.py is useless but need to change expected argv
        run_command(argv, quiet=True)

        print("Installing nfs on all running instances...")
        update_command = "sudo apt-get -y update"
        install_nfs_command = "sudo apt-get -y install nfs-common"
        create_mount_command = "mkdir %s" % configuration["nfs_mount_point"]
        
        # ~ mount the nfs filesystem
        # -t: filesystem type 
        # -o: filesystem-specific options
        setup_nfs_command = "sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 %s:/ %s" % (configuration["nfs_ip_address"], configuration["nfs_mount_point"])
        reduce_permissions_command = "sudo chmod 777 %s " % configuration["nfs_mount_point"]
        command = update_command + " && " + install_nfs_command + " && " + create_mount_command + " && " + setup_nfs_command + " && " + reduce_permissions_command

        # pretty hackish
        argv = ["python", "inception_ec2.py", live_instances_string, command]
        run_command(argv, quiet=True)
        return

    # Launch instances as specified by the configuration.
    # We also want a shared filesystem to write model checkpoints.
    # For simplicity we will have the user specify the filesystem via the config.
    def launch(argv):
        method = "spot"
        if "method" in configuration:
           method = configuration["method"]
        launch_instances()
        if method == "spot":
           wait_until_instance_request_status_fulfilled()
        wait_until_running_instances_initialized()
        print('setup nfs')
        setup_nfs(0)

    def clean_launch_and_run(argv):
        # 1. Kills all instances in region
        # 2. Kills all requests in region
        # 3. Launches requests
        # 5. Waits until launch requests have all been satisfied,
        #    printing status outputs in the meanwhile
        # 4. Checks that configuration has been satisfied
        # 5. Runs inception
        shut_everything_down(None)
        launch(None)
        return run_mxnet_grid_search(None)

    def help(hmap):
        print("Usage: python inception_ec2.py [command]")
        print("Commands:")
        for k,v in hmap.items():
            print("%s - %s" % (k,v))

    ##############################
    # tf_ec2 main starting point #
    ##############################

    command_map = {
        "launch" : launch,
        "clean_launch_and_run" : clean_launch_and_run,
        "shutdown" : shut_everything_down,
        "run_mxnet_grid_search": run_mxnet_grid_search,
        "run_mxnet_loss_curve": run_mxnet_loss_curve,
        "get_hosts": get_hosts,
        "kill_all_python" : kill_all_python,
        "list_idle_instances" : summarize_idle_instances,
        "list_running_instances" : summarize_running_instances,
        "kill_python" : kill_python,
        "run_command" : run_command,
        "setup_nfs": setup_nfs,
    }
    help_map = {
        "launch" : "Launch instances",
        "clean_launch_and_run" : "Shut everything down, launch instances, wait until requests fulfilled, check that configuration is fulfilled, and launch and run inception.",
        "shutdown" : "Shut everything down by cancelling all instance requests, and terminating all instances.",
        "list_idle_instances" : "Lists all idle instances. Idle instances are running instances not running tensorflow.",
        "list_running_instances" : "Lists all running instances.",
        "run_mxnet_grid_search": "",
        "run_mxnet_loss_curve": "",
        "setup_nfs": "",
        "kill_all_python" : "Kills python running inception training on ALL instances.",
        "kill_python" : "Kills python running inception on instances indicated by instance id string separated by ',' (no spaces).",
        "run_command" : "Runs given command on instances selcted by instance id string, separated by ','.",
    }

    if len(argv) < 2:
        help(help_map)
        sys.exit(0)

    command = argv[1]
    
    # ~ this runs the command in argv[1]
    return command_map[command](argv)

if __name__ == "__main__":
    print(cfg)
    mxnet_ec2_run(sys.argv, cfg)
