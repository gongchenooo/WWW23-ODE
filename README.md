Code for (WWW'23) To Store or Not? Online Data Selection for Federated Learning with Limited Storage
## 1. Structure
* `Code/Baselines`: Implementation of 4 categories of baselines, including (1) Random Sampling (*FIFO* and *RS*), (2) Importance Sampling (*HighLoss, Gradient Norm*), (3) Data Selection for FL (*FedBalancer, Li*) and (4) FullData Setting (*FullData*)
  * `main.py`: main function of federated learning process
  * `model.py`: base models for clients and server
  * `client.py`: definition of client class
  * `server.py`: definition of server class
  * `baseline_constants.py`: some arguments and constants
  * `utils`: other functions
    * `language_utils.py`: functions for text processing
    * `model_utils.py`: functions of batch data creation, client setup, noisy data generation, data loading
    * `torch_utils.py`: transformation between numpy.array and torch.tensor
  * `fashionmnist`: model of fashionmnist dataset
    * `LeNet.py`
    * `CNN.py`
  * `HARBOX`: model of HARBOX dataset
    * `log_reg.py`
  * `synthetic`: model of synthetic dataset
    * `log_reg.py`
* `Code/ODE`: Implementation of our proposed ODE framework, including 4 versions: (1) individual client-side data selection with accurate gradient of global data (*Dream*), (2) individual client-side data selection with estimated gradient of global data (*Estimation*), (3) server-side coordinated data storage and client-side data selection with accurate global data gradient (*Coor+Dream*), (4) server-side coordinated data storage and client-side data selection with estimated global data gradient (*Coor+Estimation*)
  * `main.py`: main function of federated learning
  * `model.py`: base models for clients and server
  * `client.py`: client class
  * `server.py`: server class
  * `baseline_constants.py`: some arguments and constants
  * `utils`: other functions
    * `model_utils.py`
    * `torch_utils.py`
  * `fashionmnist`: model of fashionmnist dataset
    * `LeNet.py`
    * `CNN.py`
  * `HARBOX`: model of HARBOX dataset
    * `log_reg.py`
  * `synthetic`: model of synthetic dataset
    * `log_reg.py`
* `Data`: the directory for data storage. If you need the data, please refer to [link](https://drive.google.com/drive/folders/1JdCOcV5XiT4RtZbqkUoIMKVfs6Ti-GLz?usp=sharing) or send [email](gongchen@sjtu.edu.cn).
  * `synthetic`
    * `data`
      * `train`: json file
      * `test`: json file
  * `fashionmist`
    * `data`
      * `train`: json file
      * `test`: json file
  * `HARBOX`
    * `data`
      * `train`: json file
      * `test`: json file
## 2. Libraries
torch 1.10.1
tqdm
PIL
pandas 1.3.4
numpy 1.20.3
json
## 3. Command
* Baselines:
  * Commands:
    * `cd Baselines`
    * `python main.py -dataset <dataset_name> -model <model_name> --num-rounds 1000 --eval-every 20 --clients-per-round <clients_per_round> --seed 0 --num-epochs 5 -lr <learning_rate>, --choosing-method <data_selection_method> --buffer-size <buffer_size>`
  * Default parameters:
    <table>
      <tr>
        <th>dataset_name</th>
        <th>model_name</th>
        <th>clients_per_round</th>
        <th>learning_rate</th>
        <th>data_selection_method</th>
        <th>buffer_size</th>
      </tr>
      <tr>
        <td>synthetic</td>
        <td>log_reg</td>
        <td>10 (10/200=5%)</td>
        <td>0.0001</td>
        <td>FIFO, HighLoss, GradientNorm, FedBalancer, Li, FullData</td>
        <td>10</td>
      </tr>
      <tr>
        <td>fashionmnist</td>
        <td>LeNet</td>
        <td>5 (5/50=10%)</td>
        <td>0.001</td>
        <td>FIFO, HighLoss, GradientNorm, FedBalancer, Li, FullData</td>
        <td>10</td>
      </tr>
      <tr>
        <td>HARBOX</td>
        <td>log_reg</td>
        <td>12 (12/120=10%)</td>
        <td>0.001</td>
        <td>FIFO, HighLoss, GradientNorm, FedBalancer, Li, FullData</td>
        <td>10</td>
      </tr>
    </table>
    
  * Examples:
    * `python main.py -dataset synthetic -model log_reg --num-rounds 1000 --eval-every 20 --clients-per-round 10 --seed 0 --num-epochs 5 -lr 0.0001 --choosing-method FIFO --buffer-size 10`
    * `python main.py -dataset fashionmnist -model LeNet --num-rounds 1000 --eval-every 20 --clients-per-round 5 --seed 0 --num-epochs 5 -lr 0.001 --choosing-method FIFO --buffer-size 10`
    * `python main.py -dataset HARBOX -model log_reg --num-rounds 1000 --eval-every 20 --clients-per-round 12 --seed 0 --num-epochs 5 -lr 0.001 --choosing-method FIFO --buffer-size 10`
* ODE: Same as commands for baselines except for data selection method
  * `data_selection_methods`: 
    * *Dream*: individual client-side data selection with accurate global data gradient
    * *Estimation1*: individual client-side data selection with global data gradient estimated by aggegating the local gradient estimators of only participating clients 
    * *Estimation2*: individual client-side data selection with global data gradient estimated by updating the global gradient estimator using the local gradient estimators of participating clients
    * *Coor+Dream*: cross-client coordinated data storage + *Dream*
    * *Coor+Estimation1*: cross-client coordinated data storage + *Estimation1*
    * *Coor+Estimation2*: cross-client coordinated data storage + *Estimation2*
  * Commands
    * `python main.py -dataset synthetic -model log_reg --num-rounds 1000 --eval-every 20 --clients-per-round 10 --seed 0 --num-epochs 5 -lr 0.0001 --choosing-method Coor+Dream --buffer-size 10`
    * `python main.py -dataset fashionmnist -model LeNet --num-rounds 1000 --eval-every 20 --clients-per-round 5 --seed 0 --num-epochs 5 -lr 0.001 --choosing-method Coor+Dream --buffer-size 10`
    * `python main.py -dataset HARBOX -model log_reg --num-rounds 1000 --eval-every 20 --clients-per-round 12 --seed 0 --num-epochs 5 -lr 0.001 --choosing-method Coor+Dream --buffer-size 10`
