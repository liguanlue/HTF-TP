import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader
import argparse
import copy
sys.path.append("../utils/")
import matplotlib.pyplot as plt
import numpy as np
from models import *
from social_utils import *
import dgl
from torch.utils.tensorboard import SummaryWriter
from visualizations import *
import yaml

parser = argparse.ArgumentParser(description='PECNet')

parser.add_argument('--num_workers', '-nw', type=int, default=0)
parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--load_file', '-lf', default="PECNET_social_model.pt")
parser.add_argument('--num_trajectories', '-nt', default=20) #number of trajectories to sample
parser.add_argument('--verbose', '-v', action='store_true')
parser.add_argument('--root_path', '-rp', default="./")
parser.add_argument('--seed', default=64, type=int, required=False, help='PyTorch random seed')


args = parser.parse_args()
set_random_seed(args.seed)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
	torch.cuda.set_device(args.gpu_index)
print(device)


checkpoint = torch.load('../saved_models/{}'.format(args.load_file), map_location=device)
hyper_params = checkpoint["hyper_params"]

print(hyper_params)

def test(test_dataset, model, best_of_n = 1):

	model.eval()
	assert best_of_n >= 1 and type(best_of_n) == int
	test_loss = 0

	with torch.no_grad():
		for i, (traj ,mask_human ,mask_scene, initial_pos, scene_b, scene_picture) in enumerate(zip(test_dataset.trajectory_batches,
														  test_dataset.mask_human,
														  test_dataset.mask_scene,
														  test_dataset.initial_pos_batches,
														  test_dataset.scene_baches,
														  test_dataset.scene_picture_baches)):
			# traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)
			type_human = traj.keys()
			type_scene = scene_b.keys()
			relation_dict = {}
			for human1 in type_human:
				for human2 in type_human:
					rela_str = human1 + '<->' + human2
					if (rela_str not in mask_human.keys()):
						continue
					relation_dict[(human1, 'adj', human2)] = mask_human[rela_str]
			for human in type_human:
				for scene in type_scene:
					rela_str1 = human + '<->' + scene
					if (rela_str1 not in mask_scene.keys()):
						continue
					relation_dict[(human, 'on', scene)] = mask_scene[rela_str1]
					# rela_str2 = scene + '<->' + human
					relation_dict[(scene, '_on', human)] = np.array(mask_scene[rela_str1])[:, [1, 0]].tolist()
			g = dgl.heterograph(relation_dict)
			g = g.to(device)

			for human in type_human:
				g.nodes[human].data['x'] = torch.DoubleTensor(traj[human][:, :hyper_params['past_length'], :]).to(
					device)
				g.nodes[human].data['y'] = torch.DoubleTensor(traj[human][:, hyper_params['past_length']:, :]).to(
					device)
				g.nodes[human].data['initial_pos'] = torch.DoubleTensor(initial_pos[human]).to(device)
				x_shape = g.nodes[human].data['x'].shape
				g.nodes[human].data['x'] = g.nodes[human].data['x'].contiguous().view(-1, x_shape[1] * x_shape[2]).to(
					device)
				g.nodes[human].data['dest'] = g.nodes[human].data['y'][:, -1, :].to(device)
				y_shape = g.nodes[human].data['y'].shape
				g.nodes[human].data['future'] = g.nodes[human].data['y'][:, :-1, :].contiguous().view(y_shape[0],-1).to(device)

			for scene in type_scene:
				g.nodes[scene].data['data'] = torch.DoubleTensor(scene_b[scene]).contiguous().view(-1, 2000).to(device)
				g.nodes[scene].data['picture'] = torch.DoubleTensor(scene_picture[scene]).to(device)



			all_l2_errors_dest = {}
			all_guesses = {}
			for index in range(best_of_n):

				# dest = None
				dest_recon = model.forward(g, type_human=type_human, type_scene=type_scene, device=device)

				dest_true, l2error_sample = {}, {}
				for human in type_human:
					dest_recon[human] = dest_recon[human].cpu().numpy()
					all_guesses.setdefault(human, []).append(dest_recon[human])
					dest_true[human] = g.nodes[human].data['dest'].cpu().numpy()
					l2error_sample[human] = np.linalg.norm(dest_recon[human] - dest_true[human], axis=1)
					all_l2_errors_dest.setdefault(human, []).append(l2error_sample[human])

			l2error_avg_dest, indices, best_guess_dest, l2error_dest = {}, {}, {}, {}
			for human in type_human:
				all_l2_errors_dest[human] = np.array(all_l2_errors_dest[human])
				all_guesses[human] = np.array(all_guesses[human])
				# average error
				l2error_avg_dest[human] = np.mean(all_l2_errors_dest[human])

				# choosing the best guess
				indices[human] = np.argmin(all_l2_errors_dest[human], axis=0)

				best_guess_dest[human] = all_guesses[human][indices[human],
										 np.arange(g.nodes[human].data['x'].shape[0]), :]

				# taking the minimum error out of all guess
				l2error_dest[human] = np.mean(np.min(all_l2_errors_dest[human], axis=0))

				best_guess_dest[human] = torch.DoubleTensor(best_guess_dest[human]).to(device)

			# using the best guess for interpolation
			predicted_future, l2error_overall_type = {}, {}
			interpolated_future = model.predict(g, generated_dest=best_guess_dest, type_human=type_human,
												type_scene=type_scene)

			for human in type_human:
				interpolated_future[human] = interpolated_future[human].cpu().numpy()
				best_guess_dest[human] = best_guess_dest[human].cpu().numpy()

				# final overall prediction
				predicted_future[human] = np.concatenate((interpolated_future[human], best_guess_dest[human]), axis=1)
				predicted_future[human] = np.reshape(predicted_future[human],
													 (-1, hyper_params['future_length'], 2))  # making sure

				# ADE error
				y = g.nodes[human].data['y'].cpu().numpy()
				l2error_overall_type[human] = np.mean(np.linalg.norm(y - predicted_future[human], axis=2))
				l2error_overall_type[human] /= hyper_params["data_scale"]
				l2error_dest[human] /= hyper_params["data_scale"]
				l2error_avg_dest[human] /= hyper_params["data_scale"]

			print('{} Test time error in destination best: {:0.3f} and mean: {:0.3f}'.format('ped', l2error_dest['ped'], l2error_avg_dest['ped']))
			print('{} Test time error overall (ADE) best: {:0.3f}'.format('ped', l2error_overall_type['ped']))

	return l2error_overall_type, l2error_dest, l2error_avg_dest

def main():
	N = args.num_trajectories #number of generated trajectories
	model = PECNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'], hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"], hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
	model = model.double().to(device)
	model.load_state_dict(checkpoint["model_state_dict"])
	root_path = '..'
	test_dataset = SocialDataset(set_name="test", b_size=hyper_params["test_b_size"],
								 t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"],
								 verbose=args.verbose, root_path=root_path)
	for traj in test_dataset.trajectory_batches:
		type_human = traj.keys()
		for human in type_human:
			traj[human] -= traj[human][:, :1, :]
			traj[human] *= hyper_params["data_scale"]

	#average ade/fde for k=20 (to account for variance in sampling)
	num_samples = 15
	average_ade, average_fde = {}, {}
	for i in range(num_samples):
		test_loss, final_point_loss_best, final_point_loss_avg = test(test_dataset, model, best_of_n = N)
		for human in test_loss.keys():
			average_ade[human] =average_ade.setdefault(human,0) + test_loss[human]
			average_fde[human] =average_fde.setdefault(human,0) + final_point_loss_best[human]

	for human in average_ade.keys():
		print("{} Average ADE: {:0.3f}".format(human, average_ade[human]/num_samples))
		print("{} Average FDE: {:0.3f}".format(human, average_fde[human]/num_samples))

main()
