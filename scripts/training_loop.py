import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import random
from torch.utils.data import DataLoader
import argparse
sys.path.append("../utils/")
# from utils.social_utils import *
# from utils.models import *
from social_utils import *
from models import *
import yaml
import dgl
from torch.utils.tensorboard import SummaryWriter
from visualizations import draw_predict

import numpy as np
import os
import pathlib

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser(description='PECNet')

parser.add_argument('--num_workers', '-nw', type=int, default=0)
parser.add_argument('--config_filename', '-cfn', type=str, default='optimal.yaml')
parser.add_argument('--save_file', '-sf', type=str, default='PECNET_social_model.pt')
parser.add_argument('--verbose', '-v', action='store_false')
parser.add_argument('--epoch', '-e',type=int, default=2000)
parser.add_argument('--seed', default=64, type=int, required=False, help='PyTorch random seed')
parser.add_argument('--run', default='pecv17_skip_d_30_notavg_nohgt_processtestdata_addimg', required=False, type=str, help='Current run name')

args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
set_random_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open("../config/" + args.config_filename, 'r') as file:
	try:
		hyper_params = yaml.load(file, Loader = yaml.FullLoader)
	except:
		hyper_params = yaml.load(file)
file.close()
print(hyper_params)

def train(train_dataset):

	model.train()
	train_loss = 0
	total_rcl, total_kld, total_adl = 0, 0, 0
	criterion = nn.MSELoss()

	# for i, (traj, mask, initial_pos) in enumerate(zip(train_dataset.trajectory_batches, train_dataset.mask_batches, train_dataset.initial_pos_batches)):
	for i, (g, type_human,type_scene, pic_b, pic_dict_b) in enumerate(zip(train_dataset.trajectory_batches,
																   train_dataset.type_human,
																   train_dataset.type_scene,
																	train_dataset.picture_batches,
																	train_dataset.picture_dict_batches)):

		scene_name = pic_b.keys()
		pic = torch.DoubleTensor(list(pic_b.values())).to(device)

		# draw_picture(g,type_human,type_scene)
		dest_recon, mu, var, interpolated_future = model.forward(g,pic,pic_dict_b,scene_name, type_human = type_human, type_scene = type_scene, device=device)

		optimizer.zero_grad()
		rcl, kld, adl, all_RCL_dest, all_ADL_traj, all_KLD = calculate_loss(g, dest_recon, mu, var, criterion, interpolated_future, type_human = type_human)
		# loss = rcl + kld * hyper_params["kld_reg"] + adl*hyper_params["adl_reg"]
		loss = all_RCL_dest + all_KLD * hyper_params["kld_reg"] + all_ADL_traj * hyper_params["adl_reg"]
		loss.backward()
		total_rcl_type, total_kld_type, total_adl_type = {}, {}, {}
		for human in type_human:
			total_rcl_type.setdefault(human, 0)
			total_rcl_type[human] += rcl[human].item()
			total_kld_type.setdefault(human,0)
			total_kld_type[human] += kld[human].item()
			total_adl_type.setdefault(human,0)
			total_adl_type[human] += adl[human].item()
		train_loss += loss.item()
		total_rcl += all_RCL_dest.item()
		total_kld += all_KLD.item()
		total_adl += all_ADL_traj.item()
		optimizer.step()

	return train_loss, total_rcl, total_kld, total_adl, total_rcl_type, total_kld_type, total_adl_type


def test(test_dataset, best_of_n = 1,args=None):
	'''Evalutes test metrics. Assumes all test data is in one batch'''

	model.eval()
	assert best_of_n >= 1 and type(best_of_n) == int
	error_avg_des_all, error_des_all, error_ade_all = {}, {}, {}
	human_num = {}
	with torch.no_grad():

		for i, (g, type_human, type_scene, pic_b, pic_dict_b) in enumerate(zip(test_dataset.trajectory_batches,
																						test_dataset.type_human,
																						test_dataset.type_scene,
																						test_dataset.picture_batches,
																						test_dataset.picture_dict_batches)):

			scene_name = pic_b.keys()
			pic = torch.DoubleTensor(list(pic_b.values())).to(device)

			all_l2_errors_dest = {}
			all_guesses = {}
			for _ in range(best_of_n):
				# dest = None
				dest_recon = model.forward(g,pic,pic_dict_b,scene_name, type_human = type_human, type_scene = type_scene, device=device)

				dest_true, l2error_sample = {}, {}
				for human in type_human:
					dest_recon[human] = dest_recon[human].cpu().numpy()
					all_guesses.setdefault(human,[]).append(dest_recon[human])
					dest_true[human] = g.nodes[human].data['dest'].cpu().numpy()
					l2error_sample[human] = np.linalg.norm(dest_recon[human] - dest_true[human], axis = 1)
					all_l2_errors_dest.setdefault(human,[]).append(l2error_sample[human])

			l2error_avg_dest, indices, best_guess_dest, l2error_dest = {}, {}, {}, {}

			for human in type_human:
				all_l2_errors_dest[human] = np.array(all_l2_errors_dest[human])
				all_guesses[human] = np.array(all_guesses[human])
				# average error
				# l2error_avg_dest[human] = np.mean(all_l2_errors_dest[human])
				l2error_avg_dest[human] = np.sum(np.mean(all_l2_errors_dest[human],axis=0))

			# choosing the best guess
				indices[human] = np.argmin(all_l2_errors_dest[human], axis = 0)
				best_guess_dest[human] = all_guesses[human][indices[human],np.arange(g.nodes[human].data['x'].shape[0]), :]

			# taking the minimum error out of all guess
			# 	l2error_dest[human] = np.mean(np.min(all_l2_errors_dest[human], axis = 0))
				l2error_dest[human] = np.sum(np.min(all_l2_errors_dest[human], axis = 0))

				best_guess_dest[human] = torch.DoubleTensor(best_guess_dest[human]).to(device)

			# using the best guess for interpolation
			predicted_future, l2error_overall_type = {}, {}
			interpolated_future = model.predict(g, generated_dest = best_guess_dest, type_human = type_human, type_scene = type_scene)

			for human in type_human:
				interpolated_future[human] = interpolated_future[human].cpu().numpy()
				best_guess_dest[human] = best_guess_dest[human].cpu().numpy()

			# final overall prediction
				predicted_future[human] = np.concatenate((interpolated_future[human], best_guess_dest[human]), axis = 1)
				predicted_future[human] = np.reshape(predicted_future[human], (-1, hyper_params['future_length'], 2)) # making sure

			# ADE error
				y = g.nodes[human].data['y'].cpu().numpy()
				l2error_overall_type[human] = np.sum(np.mean(np.linalg.norm(y - predicted_future[human], axis = 2),axis=1))
				l2error_overall_type[human] /= hyper_params["data_scale"]
				l2error_dest[human] /= hyper_params["data_scale"]
				l2error_avg_dest[human] /= hyper_params["data_scale"]

				error_ade_all[human] = error_ade_all.setdefault(human,0) + l2error_overall_type[human]
				error_des_all[human] = error_des_all.setdefault(human,0) + l2error_dest[human]
				error_avg_des_all[human] = error_avg_des_all.setdefault(human,0) + l2error_avg_dest[human]
				human_num[human] = human_num.setdefault(human,0) + predicted_future[human].shape[0]

			# draw predict
			# if l2error_overall_type['ped'] < 11:
			# 	draw_predict(g, predicted_future, hyper_params["data_scale"], args.run, type_scene, type_human,i, mean_test, pic_b, pic_dict_b)
		type_human = {'ped','skater','biker','car'}
		for human in type_human:
			error_ade_all[human] = error_ade_all[human]/human_num[human]
			error_des_all[human] = error_des_all[human]/human_num[human]
			error_avg_des_all[human] = error_avg_des_all[human]/human_num[human]
		# print('Test time error in destination best: {:0.3f} and mean: {:0.3f}'.format(error_ade_all['ped'], l2error_avg_dest['ped']))
		# print('Test time error overall (ADE) best: {:0.3f}'.format(l2error_overall_type['ped']))

	return error_ade_all, error_des_all, error_avg_des_all

model = PECNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'], hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"], hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
model = model.double().to(device)
optimizer = optim.Adam(model.parameters(), lr =  hyper_params["learning_rate"])

root_path='..'
train_dataset = SocialDataset(set_name="train", b_size=hyper_params["train_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"], data_scale = hyper_params["data_scale"], past_length = hyper_params['past_length'], verbose=args.verbose, root_path=root_path, device = device)
test_dataset = SocialDataset(set_name="test", b_size=hyper_params["test_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"] , data_scale = hyper_params["data_scale"], past_length = hyper_params['past_length'],verbose=args.verbose, root_path=root_path, device = device)
# test_ture_dataset = SocialDataset(set_name="test", b_size=hyper_params["test_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"], verbose=args.verbose, root_path='../social_pool_data')
# shift origin and scale data

# for scene in train_dataset.scene_baches: # []
# 	type_scene = scene.keys()
# 	for s_type in type_scene:
# 		scene[s_type] = np.array(scene[s_type],dtype = float)
# 		scene[s_type] -= scene[s_type][:,:1,:]
# 		scene[s_type] *= hyper_params["data_scale"]
#
# for scene in test_dataset.scene_baches: # []
# 	type_scene = scene.keys()
# 	for s_type in type_scene:
# 		scene[s_type] = np.array(scene[s_type],dtype = float)
# 		scene[s_type] -= scene[s_type][:,:1,:]
# 		scene[s_type] *= hyper_params["data_scale"]


dir_str = 'runs/' + args.run
print(dir_str)
curr_run_dir = pathlib.Path(root_path) / dir_str
curr_run_dir.mkdir(parents=True, exist_ok=True)
tb_dir = curr_run_dir / 'tb'
tb_dir.mkdir(parents=True, exist_ok=True)
wr = SummaryWriter(tb_dir, purge_step = 1)

best_test_loss = {} # start saving after this threshold
best_endpoint_loss = {}
N = hyper_params["n_values"]
# for e in range(hyper_params['num_epochs']):
ade_index = {}
fde_index = {}
for e in range(args.epoch):
	train_loss, total_rcl, total_kld, total_adl, total_rcl_type, total_kld_type, total_adl_type = train(train_dataset)
	test_loss, final_point_loss_best, final_point_loss_avg = test(test_dataset, best_of_n = N, args=args)

	for human in test_loss.keys():
		if best_test_loss.setdefault(human,200) > test_loss[human]:
			ade_index[human] = e
			best_test_loss[human] = test_loss[human]
		if best_test_loss['ped'] < 10.25:
			save_path = root_path + '/saved_models/' + args.save_file
			torch.save({
						'hyper_params': hyper_params,
						'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict()
						}, save_path)
			print("Saved model to:\n{}".format(save_path))

		if best_endpoint_loss.setdefault(human,200) > final_point_loss_best[human] :
			fde_index[human] = e
			best_endpoint_loss[human] = final_point_loss_best[human]
	print('-----------------Epoch {}   PERFORMANCE {:0.2f} ----------------'.format(e, test_loss['ped']))
	print("Train Loss", train_loss)
	print("Total RCL = {} KLD = {} ADL = {}".format(total_rcl,total_kld,total_adl))
	for human in total_rcl_type.keys():
		print("Total {0} RCL = {1} KlD = {2} ADL = {3}".format(human,total_rcl_type[human],total_kld_type[human],total_adl_type[human]))
	for human in test_loss.keys():
		print("{} Test ADE {}".format(human, test_loss[human]))
		wr.add_scalar('Test ADE ' + human, test_loss[human], e)
		print("{} Test Average FDE (Across  all samples) {}".format(human, final_point_loss_avg[human]))
		wr.add_scalar('Test Average FDE ' + human, final_point_loss_avg[human], e)
		print("{} Test Min FDE {}".format(human, final_point_loss_best[human]))
		wr.add_scalar('Test Min FDE ' + human, final_point_loss_best[human], e)
		print("{} Test Best ADE Loss So Far {}".format(human, best_test_loss[human]))
		print("{} Test Best Min FDE {}".format(human, best_endpoint_loss[human]))
