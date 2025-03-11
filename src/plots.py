import torch
import matplotlib.pyplot as plt
import numpy as np
from networkx.utils import np_random_state
from matplotlib import cbook, cm
from matplotlib.colors import LightSource

from src.loss_functions import f_loss_states, f_loss_u, f_loss_ca, f_loss_obst, f_loss_obst_tracking, f_loss_obst_dyn_tracking


def plot_trajectories(x, xbar, n_agents, text="", save=False, filename=None, T=100, obst=False, dots=False,
                      circles=False, axis=False, min_dist=1, f=5, x_=None, T_=None):
    fig = plt.figure(f)
    if x.dim()==3:
        n_traj = x.size(2)
    else:
        n_traj = 1
        x = x.unsqueeze(2)
        if x_ is not None:
           x_ =x_.unsqueeze(2)
    if obst:
        yy, xx = np.meshgrid(np.linspace(-3, 3.5, 120), np.linspace(-3, 3, 100))
        zz = xx * 0
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                zz[i, j] = f_loss_obst(torch.tensor([xx[i, j], yy[i, j], 0.0, 0.0]))
        z_min, z_max = np.abs(zz).min(), np.abs(zz).max()
        ax = fig.subplots()
        c = ax.pcolormesh(xx, yy, zz, cmap='Greens', vmin=z_min, vmax=z_max)
        # fig.colorbar(c, ax=ax)
    # plt.xlabel(r'$q_x$')
    # plt.ylabel(r'$q_y$')
    plt.title(text)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', '#90ee90', '#c20078']
    for traj in range(n_traj):
        alphatraj =max(1-0.2*traj,0)
        for i in range(n_agents):
            if T is not None:
                plt.plot(x[:T+1,4*i,traj].detach(), x[:T+1,4*i+1,traj].detach(), color=colors[i%12], linewidth=1, alpha=alphatraj)
                # plt.plot(x[T:,4*i].detach(), x[T:,4*i+1].detach(), color=colors[i%12], linestyle='dotted', linewidth=0.5)
                plt.plot(x[T:,4*i,traj].detach(), x[T:,4*i+1,traj].detach(), color='k', linewidth=0.125, linestyle='dotted', alpha=alphatraj)
            if x_ is not None:
                # Plot the grey line for `x_`
                plt.plot(x_[:T_+1, 4*i].detach(), x_[:T_+1,4*i+1].detach(), color='grey', linewidth=0.8, linestyle='-', alpha=alphatraj)

                # Plot a ball at the initial point of `x_`
                plt.plot(x_[0, 4*i].detach(), x_[0, 4*i+1].detach(), color='grey', marker='o', markersize=6, fillstyle='none', alpha=alphatraj)

        for i in range(n_agents):
            plt.plot(x[0,4*i,traj].detach(), x[0,4*i+1,traj].detach(), color=colors[i%12], marker='o', fillstyle='none', alpha=alphatraj)
            plt.plot(xbar[4*i].detach(), xbar[4*i+1].detach(), color=colors[i%12], marker='*', alpha=alphatraj)
        ax = plt.gca()
        if dots:
            for i in range(n_agents):
                for j in range(T):
                    plt.plot(x[j, 4*i,traj].detach(), x[j, 4*i+1,traj].detach(), color=colors[i%12], marker='o', alpha=alphatraj)
        if circles:
            for i in range(n_agents):
                r = min_dist/2
                # if obst:
                #     circle = plt.Circle((x[T-1, 4*i].detach(), x[T-1, 4*i+1].detach()), r, color='tab:purple', fill=False)
                # else:
                circle = plt.Circle((x[0, 4*i,traj].detach(), x[0, 4*i+1,traj].detach()), r, color=colors[i%12], alpha=alphatraj/2,
                                    zorder=10)
                ax.add_patch(circle)
        ax.axes.xaxis.set_visible(axis)
        ax.axes.yaxis.set_visible(axis)
        # TODO: add legend ( solid line: t<T/3 , dotted line> t>T/3, etc )
    if save:
        plt.title('')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_Yaxis().set_visible(False)
        plt.savefig('figures/' + filename+'_'+text+'_trajectories.png', format='png', dpi=600)
        plt.close()
    else:
        plt.grid()
        plt.show()
    return fig


def plot_trajectories_tracking(x, xref, n_agents, text="", save=False, filename=None, T=100, obst=False, dots=False,
                      circles=False, axis=False, min_dist=1, f=5, x_=None, T_=None):
    fig = plt.figure(f)
    if x.dim()==3:
        n_traj = x.size(2)
    else:
        n_traj = 1
        x = x.unsqueeze(2)
        if x_ is not None:
           x_ =x_.unsqueeze(2)
    if obst:
        yy, xx = np.meshgrid(np.linspace(-4, 4, 120), np.linspace(-4, 4, 100))
        zz = xx * 0
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                zz[i, j] = f_loss_obst_tracking(torch.tensor([xx[i, j], yy[i, j], 0.0, 0.0]))
        z_min, z_max = np.abs(zz).min(), np.abs(zz).max()
        ax = fig.subplots()
        c = ax.pcolormesh(xx, yy, zz, cmap='Greens', vmin=z_min, vmax=z_max)
        # fig.colorbar(c, ax=ax)
    # plt.xlabel(r'$q_x$')
    # plt.ylabel(r'$q_y$')
    plt.title(text)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', '#90ee90', '#c20078']
    for traj in range(n_traj):
        alphatraj =max(1-0.2*traj,0)
        for i in range(n_agents):
            if T is not None:
                # plt.plot(x[:T+1,4*i,traj].detach(), x[:T+1,4*i+1,traj].detach(), color=colors[i%12], linewidth=1, alpha=alphatraj)
                plt.plot(x[:T+1,4*i,traj].detach(), x[:T+1,4*i+1,traj].detach(), color=colors[3], linewidth=1, alpha=alphatraj)
                # plt.plot(x[T:,4*i].detach(), x[T:,4*i+1].detach(), color=colors[i%12], linestyle='dotted', linewidth=0.5)
                plt.plot(x[T:,4*i,traj].detach(), x[T:,4*i+1,traj].detach(), color='k', linewidth=0.125, linestyle='dotted', alpha=alphatraj)
            if x_ is not None:
                # Plot the grey line for `x_`
                plt.plot(x_[:T_+1, 4*i].detach(), x_[:T_+1,4*i+1].detach(), color='grey', linewidth=0.8, linestyle='-', alpha=alphatraj)

                # Plot a ball at the initial point of `x_`
                plt.plot(x_[0, 4*i].detach(), x_[0, 4*i+1].detach(), color='grey', marker='o', markersize=6, fillstyle='none', alpha=alphatraj)

        for i in range(n_agents):
            plt.plot(x[0,4*i,traj].detach(), x[0,4*i+1,traj].detach(), color=colors[i%12], marker='o', fillstyle='none', alpha=alphatraj)
            plt.plot(xref[:, 0], xref[:, 1], 'r-.', label='Optimal ref', alpha=0.5)
        ax = plt.gca()
        if dots:
            for i in range(n_agents):
                for j in range(T):
                    plt.plot(x[j, 4*i,traj].detach(), x[j, 4*i+1,traj].detach(), color=colors[i%12], marker='o', alpha=alphatraj)
        if circles:
            for i in range(n_agents):
                r = min_dist/2
                # if obst:
                #     circle = plt.Circle((x[T-1, 4*i].detach(), x[T-1, 4*i+1].detach()), r, color='tab:purple', fill=False)
                # else:
                circle = plt.Circle((x[T, 4*i,traj].detach(), x[T, 4*i+1,traj].detach()), r, color=colors[i%12], alpha=alphatraj/2,
                                    zorder=10)
                ax.add_patch(circle)
        ax.axes.xaxis.set_visible(axis)
        ax.axes.yaxis.set_visible(axis)
        # TODO: add legend ( solid line: t<T/3 , dotted line> t>T/3, etc )

    if save:
        plt.grid()
        plt.savefig('figures/' + filename+'_'+text+'_trajectories.png', format='png', dpi=600)
        plt.close()
    else:
        plt.grid()
        plt.show()
    return fig


def plot_trajectories_tracking_dyn_obs(x, xref, n_agents, pos_obs, text="", save=False, filename=None, T=100, obst=False, dots=False,
                      circles=True, axis=False, min_dist=0.5, f=5, x_=None, T_=None):
    fig = plt.figure(f)
    if x.dim()==3:
        n_traj = x.size(2)
    else:
        n_traj = 1
        x = x.unsqueeze(2)
        if x_ is not None:
           x_ =x_.unsqueeze(2)
    if obst:
        yy, xx = np.meshgrid(np.linspace(-4, 4, 120), np.linspace(-4, 4, 100))
        zz = xx * 0
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                zz[i, j] = f_loss_obst_dyn_tracking(torch.tensor([xx[i, j], yy[i, j], 0.0, 0.0]),pos_obs)
            z_min, z_max = np.abs(zz).min(), np.abs(zz).max()
            ax = fig.subplots()
            c = ax.pcolormesh(xx, yy, zz, cmap='Greens', vmin=z_min, vmax=z_max)
        # fig.colorbar(c, ax=ax)
    # plt.xlabel(r'$q_x$')
    # plt.ylabel(r'$q_y$')
    plt.title(text)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', '#90ee90', '#c20078']
    for traj in range(n_traj):
        alphatraj =max(1-0.2*traj,0)
        for i in range(n_agents):
            if T is not None:
                # plt.plot(x[:T+1,4*i,traj].detach(), x[:T+1,4*i+1,traj].detach(), color=colors[i%12], linewidth=1, alpha=alphatraj)
                plt.plot(x[:T+1,4*i,traj].detach(), x[:T+1,4*i+1,traj].detach(), color=colors[0], linewidth=1, alpha=alphatraj)
                # plt.plot(x[T:,4*i].detach(), x[T:,4*i+1].detach(), color=colors[i%12], linestyle='dotted', linewidth=0.5)
                plt.plot(x[T:,4*i,traj].detach(), x[T:,4*i+1,traj].detach(), color='k', linewidth=0.125, linestyle='dotted', alpha=alphatraj)
            if x_ is not None:
                # Plot the grey line for `x_`
                plt.plot(x_[:T_+1, 4*i].detach(), x_[:T_+1,4*i+1].detach(), color='grey', linewidth=0.8, linestyle='-', alpha=alphatraj)

                # Plot a ball at the initial point of `x_`
                plt.plot(x_[0, 4*i].detach(), x_[0, 4*i+1].detach(), color='grey', marker='o', markersize=6, fillstyle='none', alpha=alphatraj)

        for i in range(n_agents):
            plt.plot(x[0,4*i,traj].detach(), x[0,4*i+1,traj].detach(), color=colors[i%12], marker='o', fillstyle='none', alpha=alphatraj)
            plt.plot(xref[:, 0], xref[:, 1], 'r-.', label='Reference', alpha=0.5)
        ax = plt.gca()
        if dots:
            for i in range(n_agents):
                for j in range(T):
                    plt.plot(x[j, 4*i,traj].detach(), x[j, 4*i+1,traj].detach(), color=colors[i%12], marker='o', alpha=alphatraj)
        if circles:
            for i in range(n_agents):
                r = min_dist/2
                # if obst:
                #     circle = plt.Circle((x[T-1, 4*i].detach(), x[T-1, 4*i+1].detach()), r, color='tab:purple', fill=False)
                # else:
                circle = plt.Circle((x[0, 4*i,traj].detach(), x[0, 4*i+1,traj].detach()), r, color=colors[i%12], alpha=alphatraj/2,
                                    zorder=10)
                ax.add_patch(circle)
        ax.axes.xaxis.set_visible(axis)
        ax.axes.yaxis.set_visible(axis)
        # TODO: add legend ( solid line: t<T/3 , dotted line> t>T/3, etc )

    if save:
        plt.grid()
        plt.savefig('figures/figures_trk/' + filename+'_'+text+'_trajectories.png', format='png', dpi=600)
        plt.close()
    else:
        plt.grid()
        plt.show()
    return fig

def plot_eps_norm(r,gainF,tsim,gainM,eps_log,norm_log, text="", save=False, filename=None):

    time = torch.linspace(0, r.size(0)-1, r.size(0))
    plt.figure(figsize=(4 * 2, 4))
    timeUpToNow = torch.linspace(0, tsim, tsim + 1)

    plt.xlim(0, r.size(0))
    plt.plot(timeUpToNow, eps_log[:tsim + 1, :])
    ax = plt.gca()
    ax.scatter(tsim, eps_log[tsim, :].detach())
    plt.plot(timeUpToNow, norm_log[:tsim + 1, :])
    ax = plt.gca()
    ax.scatter(tsim, norm_log[tsim, :].detach())


    plt.legend([r"$\epsilon$",r"$\epsilon^{(i)}$", r"$\|x-\bar{x}\|$", r"$\|x_{t_i}-\bar{x}\|$"], loc="upper right")
    plt.title(r"$\epsilon^{(i)}=\frac{r}{\gamma(\mathcal{F} (\mathcal{M}+1))}$: %.1f" % eps_log[tsim,:].detach() + "r$\|x-\bar{x}\|$: %.1f" % norm_log[tsim,:].detach())
    if save:
        plt.savefig('figures/' + filename + '_' + text + '_eps_normX.png', format='png', dpi=600)
    else:
        plt.show()


def plot_gain_eps_r(r,gainF,tsim,gain_log,eps_log, text="", save=False, filename=None):

    time = torch.linspace(0, r.size(0)-1, r.size(0))
    plt.figure(figsize=(4 * 2, 4))
    plt.subplot(1, 2, 1)
    plt.plot(time, r.detach())
    ax = plt.gca()
    ax.scatter(tsim, r[tsim].detach())
    timeUpToNow = torch.linspace(0, tsim, tsim + 1)
    plt.plot(timeUpToNow, gain_log[:tsim + 1, :])
    ax = plt.gca()
    ax.scatter(tsim, gain_log[tsim, :].detach())
    plt.xlabel(r'time')
    plt.title(r"evolution of $r^{(i)}$: %.1f " % r[tsim].detach() + r"and $\gamma(\mathcal{M}^{(i)})$: %.1f " % gain_log[tsim, :].detach())
    plt.legend([r'$r$',r'$r^{(i)}$',r'$\gamma(\mathcal{M})$',r'$\gamma(\mathcal{M}^{(i)})$'], loc="upper right")

    plt.subplot(1, 2, 2)
    plt.xlim(0, r.size(0))
    plt.plot(timeUpToNow, eps_log[:tsim + 1, :])
    ax = plt.gca()
    ax.scatter(tsim, eps_log[tsim, :].detach())


    plt.legend([r"$\|x-\bar{x}\|$", r"$\|x(t)-\bar{x}\|$",r"$\frac{\|r\|}{\gamma(\mathcal{F}) (\gamma(\mathcal{M})+1)}$"], loc="upper right")
    plt.title(r"$\epsilon^{(i)}=\| x(t)-\bar{x} \|$: %.1f" % eps_log[tsim,:].detach())
    if save:
        plt.savefig('figures/' + filename + '_' + text + '_r_gainM.png', format='png', dpi=600)
    else:
        plt.show()


#########
# This function plots the trajectories of agents' positions and velocities over time. It takes in the following arguments:
#t_end: the end time of the simulation
#n_agents: the number of agents in the simulation
#x: a tensor of shape (t_end, 4*n_agents) containing the positions and velocities of all agents over time
#u: an optional tensor of shape (t_end, 2*n_agents) containing the control inputs for all agents over time
#text: an optional string to add as a title to the plot
#save: a boolean indicating whether to save the plot as a file or display it
#filename: an optional string to use as the filename if save is True
#The function first creates a time tensor t and determines the number of subplots to create based on whether u is provided. It then creates a figure with the appropriate number of subplots and plots the positions and velocities of all agents over time in the first two subplots. If u is provided, it plots the control inputs for all agents over time in the third subplot. Finally, it adds a title to the entire figure and saves it as a file if save is True, or displays it if save is False.
def plot_traj_vs_time(t_end, n_agents, x, u=None, text="", save=False, filename=None):
    t = torch.linspace(0,t_end-1, t_end)
    if u is not None:
        p = 4
    else:
        p = 3
    plt.figure(figsize=(4*p, 4))
    plt.subplot(1, p, 1)
    for i in range(n_agents):
        plt.plot(t, x[:,4*i])
    plt.xlabel(r'$t$')
    plt.title(r'$position \ x(t)$')
    plt.grid()
    plt.subplot(1, p, 2)
    for i in range(n_agents):
        plt.plot(t, x[:,4*i+1])
    plt.xlabel(r'$t$')
    plt.title(r'$position \ y(t)$')
    plt.grid()
    plt.subplot(1, p, 3)
    for i in range(n_agents):
        plt.plot(t, x[:,4*i+2])
        plt.plot(t, x[:,4*i+3])
    plt.xlabel(r'$t$')
    plt.title(r'velocity \ v(t)')
    plt.grid()
    plt.suptitle(text)
    if p == 4:
        plt.subplot(1, 4, 4)
        for i in range(n_agents):
            plt.plot(t, u[:, 2*i])
            plt.plot(t, u[:, 2*i+1])
        plt.xlabel(r'$t$')
        plt.grid()
        plt.title(r'$control \ action \ u(t)$')
    if save:
        plt.savefig('figures/' + filename + '_' + text + '_x_u.eps', format='eps')
    else:
        plt.show()

def plot_cost(x,u,alpha_u,alpha_ca,alpha_obst,sys,Q,min_dist):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    loss_x = 5 * f_loss_states(t, X, sys, Q)
    loss_u = alpha_u * f_loss_u(t, u)
    loss_ca = alpha_ca * f_loss_ca(x, sys, min_dist)
    if alpha_obst != 0:
        loss_obst = alpha_obst * f_loss_obst(x)
    loss = loss_x + loss_u + loss_ca + loss_obst
    Z = loss

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.grid()
    plt.show()


def plot_losses(epochs, lossl, lossxl, lossul, losscal, lossobstl, text="", save=False, filename=None):
    t = torch.linspace(0, epochs - 1, epochs)
    plt.figure(figsize=(4 * 2, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t, lossl[:])
    plt.grid()
    plt.xlabel(r'$epoch$')
    plt.title(r'$loss$')
    plt.subplot(1, 2, 2)
    plt.plot(t, lossxl[:])
    plt.grid()
    plt.xlabel(r'$epoch$')
    plt.title(r'$lossx$')

    plt.figure(figsize=(4 * 3, 4))
    plt.subplot(1, 3, 1)
    plt.plot(t, lossul[:])
    plt.grid()
    plt.xlabel(r'$epoch$')
    plt.title(r'$lossu$')
    plt.subplot(1, 3, 2)
    plt.plot(t, losscal[:])
    plt.grid()
    plt.xlabel(r'$epoch$')
    plt.title(r'$lossoa$')
    plt.suptitle(text)
    plt.subplot(1, 3, 3)
    plt.plot(t, lossobstl[:])
    plt.suptitle(text)
    plt.grid()
    plt.xlabel(r'$t$')
    plt.title(r'$lossobst$')

    if save:
        plt.savefig('figures/' + filename + '_' + text + '_x_u.eps', format='eps')
    else:
        plt.show()

def plot_losses_tracking(epochs, lossl, lossxl, lossul, lossobstl, text="", save=False, filename=None):
    t = torch.linspace(0, epochs - 1, epochs)
    plt.figure(figsize=(4 * 2, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t, lossl[:])
    plt.grid()
    plt.xlabel(r'$epoch$')
    plt.title(r'$loss$')
    plt.subplot(1, 2, 2)
    plt.plot(t, lossxl[:])
    plt.grid()
    plt.xlabel(r'$epoch$')
    plt.title(r'$lossx$')

    plt.figure(figsize=(4 * 3, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t, lossul[:])
    plt.grid()
    plt.xlabel(r'$epoch$')
    plt.title(r'$lossu$')
    plt.subplot(1, 2, 2)
    plt.plot(t, lossobstl[:])
    plt.suptitle(text)
    plt.grid()
    plt.xlabel(r'$t$')
    plt.title(r'$lossobst$')

    if save:
        plt.savefig('figures/' + filename + '_' + text + 'tracking_x_u.eps', format='eps')
    else:
        plt.show()


