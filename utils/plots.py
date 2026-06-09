import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from matplotlib.ticker import ScalarFormatter
from matplotlib import rcParams
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba

class Plots:                             
    
    def __init__(self) -> None:
        """
        Initializes the Plots class and sets global plotting parameters, including dark mode and font settings.
        """

        rcParams["font.family"] = "Arial"

        global dark_mode_images
        dark_mode_images = True
        global dark_mode_plots 
        dark_mode_plots = True
        global arrows_instead_frame
        arrows_instead_frame = True

        if(dark_mode_images == True):
            self.text_color_images = "white"
            self.background_color_images = "black"
        else:
            self.text_color_images = "black"
            self.background_color_images = "white"

        if(dark_mode_plots == True):
            self.text_color_plot = "white"
            self.background_color_plot = "black"
        else:
            self.text_color_plot = "black"
            self.background_color_plot = "white"

    def show_image(self, img, pixel_size_um, mark_bad, ax, fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, title, cbar_join=False, vmin = None, vmax = None, EI_aspect = None, colormap=plt.cm.gist_gray.copy()):
        """
        Displays an image with a scalebar, colorbar, and custom formatting.

        Args:
            img (np.ndarray): Image to display.
            pixel_size_um (float): Pixel size in micrometers.
            mark_bad (bool): Whether to mark bad pixels.
            ax (matplotlib.axes.Axes): Axis to plot on.
            fig (matplotlib.figure.Figure): Figure object.
            scalebar_color (str): Color of the scalebar.
            scalebar_pad (float): Padding for the scalebar.
            scalebar_alpha (float): Alpha transparency for the scalebar box.
            scalebar_fontsize (int): Font size for the scalebar.
            title (str): Title for the image.
            cbar_join (bool, optional): If True, join colorbar with other axes.
            vmin, vmax (float, optional): Color limits.
            EI_aspect (float, optional): Aspect ratio for EI images.
            colormap (matplotlib.colors.Colormap, optional): Colormap to use.

        Returns:
            matplotlib.image.AxesImage: The displayed image object.
        """

        #colormap = plt.cm.gist_gray.copy()

        if mark_bad:
            colormap.set_bad(color='red')
        else:
            colormap.set_bad(color="white")
    
        #fig.patch.set_alpha(0)  # Transparent canvas

        if((vmin != None) and (vmax == None)):
            im = ax.imshow(img, cmap=colormap, interpolation='auto', vmin=vmin)
        elif((vmin == None) and (vmax != None)):
            im = ax.imshow(img, cmap=colormap, interpolation='auto', vmax=vmax)
        elif((vmin != None) and (vmax != None)):
            im = ax.imshow(img, cmap=colormap, interpolation='auto', vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(img, cmap=colormap, interpolation='auto')

        if(EI_aspect != None):
            ax.set_aspect(EI_aspect/2)
    
        ax.set_title(title,color=self.text_color_images, fontsize=10)
    
        if(pixel_size_um != None):
            scalebar = ScaleBar(pixel_size_um, units='µm', length_fraction=0.2, location="lower right",
                                box_alpha=scalebar_alpha, pad=scalebar_pad, color=scalebar_color,
                                font_properties={"size": scalebar_fontsize}, box_color="black")
        else:
            scalebar = ScaleBar(1, units="px", dimension="pixel-length", length_fraction=0.2, location="lower right",
                                box_alpha=scalebar_alpha, pad=scalebar_pad, color=scalebar_color,
                                font_properties={"size": scalebar_fontsize}, box_color="black")
        ax.add_artist(scalebar)
    
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_linewidth(0.75)  # Set to desired width in pts
            spine.set_edgecolor(self.background_color_images)
    
        # Colorbar specific to this axis
        if(cbar_join == False):
            # Colorbar specific to this axis
            cbar = fig.colorbar(im, ax=ax) # shrink reduces length; aspect adjusts     width
            for spine in cbar.ax.spines.values():
                spine.set_edgecolor(self.text_color_images)
            cbar.ax.tick_params(colors=self.text_color_images)
        
            font_prop = fm.FontProperties(size=10)
            for tick_label in cbar.ax.get_yticklabels():
                tick_label.set_fontproperties(font_prop)

            # Set scientific formatter on colorbar
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-1, 1))
            cbar.ax.yaxis.set_major_formatter(formatter)
    
            # Optional: increase font size or adjust offset label
            cbar.ax.yaxis.get_offset_text().set_size(8)
            cbar.ax.yaxis.get_offset_text().set_va('bottom')
            # Center the ×10ⁿ label horizontally
            cbar.ax.yaxis.get_offset_text().set_horizontalalignment('center')
            # Move it to the right place (under the colorbar)
            x = 2  # middle of colorbar (in axis fraction coords)
            cbar.ax.yaxis.get_offset_text().set_x(x)

    
        fig.patch.set_facecolor(self.background_color_images)
    
        return im

    
    def show_plot(self, xdata, ydata, ax, fig, title, label):
        """
        Plots x-y data with custom formatting, legend, and axis arrows.

        Args:
            xdata (array-like): X-axis data.
            ydata (array-like): Y-axis data.
            ax (matplotlib.axes.Axes): Axis to plot on.
            fig (matplotlib.figure.Figure): Figure object.
            title (str): Title for the plot.
            label (str): Label for the data series.
        """

        im = ax.plot(xdata, ydata, label = label, linestyle='--', marker='o', markersize=5)
        ax.legend(prop={'size': 10})

        ax.set_title(title,color=self.text_color_plot, fontsize=10)

        ax.set_xticks(xdata)

        if(arrows_instead_frame == False):
            for spine in ax.spines.values():
                spine.set_linewidth(0.75)  # Set to desired width in pts
                spine.set_edgecolor(self.background_color_images)
                spine.set_color(self.text_color_plot)
        else:
            # Hide the default spines
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(False)
            
            # Keep ticks visible
            ax.tick_params(axis='both', which='both',
                           direction='out',  # or 'inout' as you prefer
                           length=4, width=1,
                           bottom=True, top=False, left=True, right=False)

            
            # Get limits for arrow placement
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            # Add arrows for x and y axes
            arrowprops = dict(arrowstyle='->', linewidth=0.75, color=self.text_color_plot, shrinkA=0, shrinkB=0)
            # shrinkA=0, shrinkB=0 helps to connect arrows by its tails
            #ax.annotate('', xy=(x_max, y_min), xytext=(x_min, y_min), arrowprops=arrowprops)  # x-axis
            #ax.annotate('', xy=(x_min,y_max), xytext=(x_min, y_min), arrowprops=arrowprops)  # y-axis
            # Draw x-axis arrow (from left to right)
            ax.annotate('', xy=(1.05, 0), xytext=(0, 0), xycoords='axes fraction', arrowprops=arrowprops)
            # Draw y-axis arrow (from bottom to top)
            ax.annotate('', xy=(0, 1.05), xytext=(0, 0), xycoords='axes fraction', arrowprops=arrowprops)

        ax.tick_params(colors=self.text_color_plot)
        ax.xaxis.label.set_color(self.text_color_plot)
        ax.yaxis.label.set_color(self.text_color_plot)
        ax.set_facecolor(self.background_color_plot)

        for label in ax.get_xticklabels():
            label.set_fontsize(10)

        for label in ax.get_yticklabels():
            label.set_fontsize(10)

        # Use scientific notation on y-axis with single global multiplier
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-1, 1))  # Always use scientific if outside this range
        ax.yaxis.set_major_formatter(formatter)
        
        # Show global multiplier (e.g., ×10^5)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-1, 1))
        ax.yaxis.get_offset_text().set_size(10)  # Optional: control font size
        ax.yaxis.get_offset_text().set_va('bottom')  # Optional: adjust vertical alignment
    
        fig.patch.set_facecolor(self.background_color_images)

    def layout_plot(self, xdatas, ydatas, dict_params):
        """
        Arranges multiple plots in a grid layout.

        Args:
            xdatas (list): List of x-axis data arrays.
            ydatas (list): List of y-axis data arrays.
            dict_params (dict): Dictionary with layout parameters.

        Returns:
            matplotlib.figure.Figure: The figure object containing the layout.
        """

        n_subplots = dict_params["n_subplots"]
        plot_size = dict_params["plot_size"]
        titles = dict_params["titles"]
        plots_space = dict_params["plots_space"]
        labels = dict_params["labels"]

        fig, axes = plt.subplots(n_subplots[0], n_subplots[1], figsize=(plot_size[0]/72, plot_size[1]/72), gridspec_kw={'wspace': plots_space[0],'hspace': plots_space[1]})  
        #fig.patch.set_alpha(0)  # Set the figure background to transparent if needed
        
        # Flatten the axes array for easier iteration
        if(n_subplots[0]*n_subplots[1] != 1):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        
        for i in range(len(ydatas)):
            self.show_plot(xdatas[i], ydatas[i], ax=axes[0], fig=fig, title=titles[0], label=labels[i])
    
        fig.patch.set_facecolor(self.background_color_plot)
    
        return fig
    
    def show_bars(self, xdata, ydata, ax, fig, title, label):
        """
        Plots bar data with custom formatting and axis arrows.

        Args:
            xdata (array-like): X-axis data.
            ydata (array-like): Y-axis data.
            ax (matplotlib.axes.Axes): Axis to plot on.
            fig (matplotlib.figure.Figure): Figure object.
            title (str): Title for the bar plot.
            label (str): Label for the data series.
        """

        # Calculate widths as differences between consecutive x
        widths = np.diff(xdata)

        # For the last bar, repeat the last width (or assign default)
        last_width = widths[-1]  # or choose some fixed value like 1.0
        widths = np.append(widths, last_width)

        im = ax.bar(xdata, ydata, width=widths, align='edge')  # align bars to the left edge
        #ax.legend(prop={'size': 10})

        ax.set_title(title,color=self.text_color_plot, fontsize=10)

        if(len(xdata) < 10):
            ax.set_xticks(xdata)
        else:
            tick_positions = np.linspace(xdata.min(), xdata.max(), 10)
            ax.set_xticks(tick_positions)

        if(arrows_instead_frame == False):
            for spine in ax.spines.values():
                spine.set_linewidth(0.75)  # Set to desired width in pts
                spine.set_edgecolor(self.background_color_images)
                spine.set_color(self.text_color_plot)
        else:
            # Hide the default spines
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(False)
            
            # Keep ticks visible
            ax.tick_params(axis='both', which='both',
                           direction='out',  # or 'inout' as you prefer
                           length=4, width=1,
                           bottom=True, top=False, left=True, right=False)

            
            # Get limits for arrow placement
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            # Add arrows for x and y axes
            arrowprops = dict(arrowstyle='->', linewidth=0.75, color=self.text_color_plot, shrinkA=0, shrinkB=0)
            # shrinkA=0, shrinkB=0 helps to connect arrows by its tails
            #ax.annotate('', xy=(x_max, y_min), xytext=(x_min, y_min), arrowprops=arrowprops)  # x-axis
            #ax.annotate('', xy=(x_min,y_max), xytext=(x_min, y_min), arrowprops=arrowprops)  # y-axis
            # Draw x-axis arrow (from left to right)
            ax.annotate('', xy=(1.05, 0), xytext=(0, 0), xycoords='axes fraction', arrowprops=arrowprops)
            # Draw y-axis arrow (from bottom to top)
            ax.annotate('', xy=(0, 1.05), xytext=(0, 0), xycoords='axes fraction', arrowprops=arrowprops)

        ax.tick_params(colors=self.text_color_plot)
        ax.xaxis.label.set_color(self.text_color_plot)
        ax.yaxis.label.set_color(self.text_color_plot)
        ax.set_facecolor(self.background_color_plot)

        for label in ax.get_xticklabels():
            label.set_fontsize(10)

        for label in ax.get_yticklabels():
            label.set_fontsize(10)

        # Use scientific notation on y-axis with single global multiplier
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-1, 1))  # Always use scientific if outside this range
        ax.yaxis.set_major_formatter(formatter)
        
        # Show global multiplier (e.g., ×10^5)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-1, 1))
        ax.yaxis.get_offset_text().set_size(10)  # Optional: control font size
        ax.yaxis.get_offset_text().set_va('bottom')  # Optional: adjust vertical alignment
    
        fig.patch.set_facecolor(self.background_color_images)

    def layout_bars(self, xdatas, ydatas, dict_params):
        """
        Arranges multiple bar plots in a grid layout.

        Args:
            xdatas (list): List of x-axis data arrays.
            ydatas (list): List of y-axis data arrays.
            dict_params (dict): Dictionary with layout parameters.

        Returns:
            matplotlib.figure.Figure: The figure object containing the layout.
        """

        n_subplots = dict_params["n_subplots"]
        plot_size = dict_params["plot_size"]
        titles = dict_params["titles"]
        plots_space = dict_params["plots_space"]
        labels = dict_params["labels"]
        axis_labels = dict_params["axis_labels"]

        fig, axes = plt.subplots(n_subplots[0], n_subplots[1], figsize=(plot_size[0]/72, plot_size[1]/72), gridspec_kw={'wspace': plots_space[0],'hspace': plots_space[1]})  
        #fig.patch.set_alpha(0)  # Set the figure background to transparent if needed
        
        # Flatten the axes array for easier iteration
        if(n_subplots[0]*n_subplots[1] != 1):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        
        for i in range(len(ydatas)):
            self.show_bars(xdatas[i], ydatas[i], ax=axes[i], fig=fig, title=titles[i], label=labels[i])
            axes[i].set_xlabel(axis_labels[1])
            axes[i].set_ylabel(axis_labels[0])
    
        fig.patch.set_facecolor(self.background_color_plot)
    
        return fig
    
    def show_hist(self, data, ax, fig, title, label):
        """
        Plots a histogram of the data with custom formatting and axis arrows.

        Args:
            data (array-like): Data to plot as a histogram.
            ax (matplotlib.axes.Axes): Axis to plot on.
            fig (matplotlib.figure.Figure): Figure object.
            title (str): Title for the histogram.
            label (str): Label for the data series.
        """

        im = ax.hist(data, bins=50, label=label, alpha=0.5)  # align bars to the left edge
        ax.legend(prop={'size': 10})

        ax.set_title(title,color=self.text_color_plot, fontsize=10)

        if(arrows_instead_frame == False):
            for spine in ax.spines.values():
                spine.set_linewidth(0.75)  # Set to desired width in pts
                spine.set_edgecolor(self.background_color_images)
                spine.set_color(self.text_color_plot)
        else:
            # Hide the default spines
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(False)
            
            # Keep ticks visible
            ax.tick_params(axis='both', which='both',
                           direction='out',  # or 'inout' as you prefer
                           length=4, width=1,
                           bottom=True, top=False, left=True, right=False)

            
            # Get limits for arrow placement
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            # Add arrows for x and y axes
            arrowprops = dict(arrowstyle='->', linewidth=0.75, color=self.text_color_plot, shrinkA=0, shrinkB=0)
            # shrinkA=0, shrinkB=0 helps to connect arrows by its tails
            #ax.annotate('', xy=(x_max, y_min), xytext=(x_min, y_min), arrowprops=arrowprops)  # x-axis
            #ax.annotate('', xy=(x_min,y_max), xytext=(x_min, y_min), arrowprops=arrowprops)  # y-axis
            # Draw x-axis arrow (from left to right)
            ax.annotate('', xy=(1.05, 0), xytext=(0, 0), xycoords='axes fraction', arrowprops=arrowprops)
            # Draw y-axis arrow (from bottom to top)
            ax.annotate('', xy=(0, 1.05), xytext=(0, 0), xycoords='axes fraction', arrowprops=arrowprops)

        ax.tick_params(colors=self.text_color_plot)
        ax.xaxis.label.set_color(self.text_color_plot)
        ax.yaxis.label.set_color(self.text_color_plot)
        ax.set_facecolor(self.background_color_plot)

        for label in ax.get_xticklabels():
            label.set_fontsize(10)

        for label in ax.get_yticklabels():
            label.set_fontsize(10)

        # Use scientific notation on y-axis with single global multiplier
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-1, 1))  # Always use scientific if outside this range
        ax.yaxis.set_major_formatter(formatter)
        
        # Show global multiplier (e.g., ×10^5)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-1, 1))
        ax.yaxis.get_offset_text().set_size(10)  # Optional: control font size
        ax.yaxis.get_offset_text().set_va('bottom')  # Optional: adjust vertical alignment
    
        fig.patch.set_facecolor(self.background_color_images)

    def layout_hist(self, datas, dict_params):
        """
        Arranges multiple histograms in a grid layout.

        Args:
            datas (list): List of data arrays for histograms.
            dict_params (dict): Dictionary with layout parameters.

        Returns:
            matplotlib.figure.Figure: The figure object containing the layout.
        """

        n_subplots = dict_params["n_subplots"]
        plot_size = dict_params["plot_size"]
        titles = dict_params["titles"]
        plots_space = dict_params["plots_space"]
        labels = dict_params["labels"]
        axis_labels = dict_params["axis_labels"]

        fig, axes = plt.subplots(n_subplots[0], n_subplots[1], figsize=(plot_size[0]/72, plot_size[1]/72), gridspec_kw={'wspace': plots_space[0],'hspace': plots_space[1]})  
        #fig.patch.set_alpha(0)  # Set the figure background to transparent if needed
        
        # Flatten the axes array for easier iteration
        if(n_subplots[0]*n_subplots[1] != 1):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i in range(len(datas)):
            self.show_hist(datas[i], ax=axes[0], fig=fig, title=titles[0], label=labels[i])
            axes[0].set_xlabel(axis_labels[1])
            axes[0].set_ylabel(axis_labels[0])
    
        fig.patch.set_facecolor(self.background_color_plot)
    
        return fig
    
    def plot_setup_ei(self, dict_params):
        """
        Plots a schematic of the Edge Illumination (EI) X-ray setup.

        Args:
            dict_params (dict): Dictionary of setup parameters.

        Returns:
            matplotlib.figure.Figure: The figure object containing the setup schematic.
        """
       
        self.type_of_source = dict_params["Source geometry"]
        self.binning_factor = int(float(dict_params["Binning factor"]))
        self.sim_pixel_m = float(dict_params["Sim. pixel (μm)"])*1e-6/self.binning_factor
        self.px_um = float(dict_params["Period (μm)"]) # Period in um
        self.M_mask = 2*self.sim_pixel_m*self.binning_factor/(self.px_um*1e-6) 
        self.shift_z_mask_cm = float(dict_params["Shift grating in prop. axis (cm)"])
        self.d_source_det = float(dict_params["Source-Detector distance (m)"])
        self.d_source_grat = self.d_source_det/self.M_mask + (self.shift_z_mask_cm*1e-2)
        self.d_source_samp = self.d_source_grat + float(dict_params["Grating-Sample distance (m)"])
        self.d_grat_det = self.d_source_det - self.d_source_grat
        self.d_samp_det = self.d_source_det - self.d_source_samp
        self.d_grat_samp = np.abs(self.d_source_samp - self.d_source_grat)
        self.d_prop = self.d_samp_det - (float(dict_params["Whole sample thickness (mm)"])*1e-3)/2
        lightorange = to_rgba("orange", alpha=0.8)
 
         # Create a figure and axis
        fig, ax = plt.subplots()
 
        # Create a Rectangle object
        source = Rectangle((-0.03, -0.10), width=0.03, height=0.3, linewidth=2, edgecolor='blue', facecolor='lightblue')
        detector = Rectangle((self.d_source_det, -0.3), width=0.01, height=0.6, linewidth=2, edgecolor='green',  facecolor='lightgreen')
        grating = Rectangle((self.d_source_grat-0.01/2, -0.3), width=0.01, height=0.6, linewidth=2, edgecolor='orange',  facecolor=lightorange)
        sample = Rectangle((self.d_source_samp-(float(dict_params["Whole sample thickness (mm)"])*1e-3)/2, -0.3), width=float(dict_params["Whole sample thickness (mm)"])*1e-3, height=0.6, linewidth=2,  edgecolor='red', facecolor='pink')
 
        # Create a Rectangle object legend
        source_legend = Rectangle((0.5*self.d_source_det/8, -0.7), width=self.d_source_det/8, height=0.03, linewidth=2,  edgecolor='blue', facecolor='lightblue')
        detector_legend = Rectangle((2.5*self.d_source_det/8, -0.7), width=self.d_source_det/8, height=0.03, linewidth=2,  edgecolor='green', facecolor='lightgreen')
        grating_legend = Rectangle((4.5*self.d_source_det/8, -0.7), width=self.d_source_det/8, height=0.03, linewidth=2,  edgecolor='orange', facecolor=lightorange)
        sample_legend = Rectangle((6.5*self.d_source_det/8, -0.7), width=self.d_source_det/8, height=0.03, linewidth=2, edgecolor='red',  facecolor='pink')
 
        # Add the rectangle to the plot
        ax.add_patch(source)
        ax.add_patch(detector)
        ax.add_patch(grating)
        ax.add_patch(sample)
 
        ax.add_patch(source_legend)
        ax.add_patch(detector_legend)
        ax.add_patch(grating_legend)
        ax.add_patch(sample_legend)
 
        ax.text(1*self.d_source_det/8, -0.76, 'Source', color='lightblue', ha='center', va='center', fontsize=10)
        ax.text(3*self.d_source_det/8, -0.76, 'Detector', color='lightgreen', ha='center', va='center', fontsize=10)
        ax.text(5*self.d_source_det/8, -0.76, "Grating" , color=lightorange, ha='center', va='center', fontsize=10)
        ax.text(7*self.d_source_det/8, -0.76, "Sample" , color='pink', ha='center', va='center', fontsize=10)
 
        # Set the limits of the plot
        ax.set_xlim(-0.15, self.d_source_det + 0.15)
        ax.set_ylim(-1, 1)
 
        ax.set_xlabel("Propagation axis (m)")
 
        ax.set_yticks([])

        if(arrows_instead_frame == False):
            for spine in ax.spines.values():
                spine.set_linewidth(0.75)  # Set to desired width in pts
                spine.set_edgecolor(self.background_color_images)
                spine.set_color(self.text_color_plot)
        else:
            # Hide the default spines
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(False)
            
            # Keep ticks visible
            ax.tick_params(axis='both', which='both',
                           direction='out',  # or 'inout' as you prefer
                           length=4, width=1,
                           bottom=True, top=False, left=True, right=False)

            
            # Get limits for arrow placement
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            # Add arrows for x and y axes
            arrowprops = dict(arrowstyle='->', linewidth=0.75, color=self.text_color_plot, shrinkA=0, shrinkB=0)
            # shrinkA=0, shrinkB=0 helps to connect arrows by its tails
            #ax.annotate('', xy=(x_max, y_min), xytext=(x_min, y_min), arrowprops=arrowprops)  # x-axis
            #ax.annotate('', xy=(x_min,y_max), xytext=(x_min, y_min), arrowprops=arrowprops)  # y-axis
            # Draw x-axis arrow (from left to right)
            ax.annotate('', xy=(1.05, 0), xytext=(0, 0), xycoords='axes fraction', arrowprops=arrowprops)

        ax.tick_params(colors=self.text_color_plot)
        ax.xaxis.label.set_color(self.text_color_plot)
        ax.set_facecolor(self.background_color_plot)

        for label in ax.get_xticklabels():
            label.set_fontsize(10)

        for label in ax.get_yticklabels():
            label.set_fontsize(10)

        ax.set_title("X-ray Setup", color=self.text_color_plot, fontsize=10)
    
        fig.patch.set_facecolor(self.background_color_images)
 
        # Show the plot
        #plt.show()

        return fig
    
    def plot_setup_sbi(self, dict_params):
        """
        Plots a schematic of the Sandpaper-based Imaging (SBI) X-ray setup.

        Args:
            dict_params (dict): Dictionary of setup parameters.

        Returns:
            matplotlib.figure.Figure: The figure object containing the setup schematic.
        """
       
        self.type_of_source = dict_params["Source geometry"]
        self.binning_factor = int(float(dict_params["Binning factor"]))
        self.sim_pixel_m = float(dict_params["Sim. pixel (μm)"])*1e-6/self.binning_factor
        self.d_source_det = float(dict_params["Source-Detector distance (m)"])
        self.d_source_grat = float(dict_params["Source-Grating distance (m)"])
        self.d_source_samp = self.d_source_grat + float(dict_params["Grating-Sample distance (m)"])
        self.d_grat_det = self.d_source_det - self.d_source_grat
        self.d_samp_det = self.d_source_det - self.d_source_samp
        self.d_grat_samp = np.abs(self.d_source_samp - self.d_source_grat)
        self.d_prop = self.d_samp_det - (float(dict_params["Whole sample thickness (mm)"])*1e-3)/2
        lightwhite = to_rgba("white", alpha=0.8)
 
         # Create a figure and axis
        fig, ax = plt.subplots()
 
        # Create a Rectangle object
        source = Rectangle((-0.03, -0.10), width=0.03, height=0.3, linewidth=2, edgecolor='blue', facecolor='lightblue')
        detector = Rectangle((self.d_source_det, -0.3), width=0.01, height=0.6, linewidth=2, edgecolor='green',  facecolor='lightgreen')
        grating = Rectangle((self.d_source_grat-0.01/2, -0.3), width=0.01, height=0.6, linewidth=2, edgecolor='white',  facecolor=lightwhite)
        sample = Rectangle((self.d_source_samp-(float(dict_params["Whole sample thickness (mm)"])*1e-3)/2, -0.3), width=float(dict_params["Whole sample thickness (mm)"])*1e-3, height=0.6, linewidth=2,  edgecolor='red', facecolor='pink')
 
        # Create a Rectangle object legend
        source_legend = Rectangle((0.5*self.d_source_det/8, -0.7), width=self.d_source_det/8, height=0.03, linewidth=2,  edgecolor='blue', facecolor='lightblue')
        detector_legend = Rectangle((2.5*self.d_source_det/8, -0.7), width=self.d_source_det/8, height=0.03, linewidth=2,  edgecolor='green', facecolor='lightgreen')
        grating_legend = Rectangle((4.5*self.d_source_det/8, -0.7), width=self.d_source_det/8, height=0.03, linewidth=2,  edgecolor='white', facecolor=lightwhite)
        sample_legend = Rectangle((6.5*self.d_source_det/8, -0.7), width=self.d_source_det/8, height=0.03, linewidth=2, edgecolor='red',  facecolor='pink')
 
        # Add the rectangle to the plot
        ax.add_patch(source)
        ax.add_patch(detector)
        ax.add_patch(grating)
        ax.add_patch(sample)
 
        ax.add_patch(source_legend)
        ax.add_patch(detector_legend)
        ax.add_patch(grating_legend)
        ax.add_patch(sample_legend)
 
        ax.text(1*self.d_source_det/8, -0.76, 'Source', color='lightblue', ha='center', va='center', fontsize=10)
        ax.text(3*self.d_source_det/8, -0.76, 'Detector', color='lightgreen', ha='center', va='center', fontsize=10)
        ax.text(5*self.d_source_det/8, -0.76, "Sandpaper" , color=lightwhite, ha='center', va='center', fontsize=10)
        ax.text(7*self.d_source_det/8, -0.76, "Sample" , color='pink', ha='center', va='center', fontsize=10)
 
        # Set the limits of the plot
        ax.set_xlim(-0.15, self.d_source_det + 0.15)
        ax.set_ylim(-1, 1)
 
        ax.set_xlabel("Propagation axis (m)")
 
        ax.set_yticks([])

        if(arrows_instead_frame == False):
            for spine in ax.spines.values():
                spine.set_linewidth(0.75)  # Set to desired width in pts
                spine.set_edgecolor(self.background_color_images)
                spine.set_color(self.text_color_plot)
        else:
            # Hide the default spines
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(False)
            
            # Keep ticks visible
            ax.tick_params(axis='both', which='both',
                           direction='out',  # or 'inout' as you prefer
                           length=4, width=1,
                           bottom=True, top=False, left=True, right=False)

            
            # Get limits for arrow placement
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            # Add arrows for x and y axes
            arrowprops = dict(arrowstyle='->', linewidth=0.75, color=self.text_color_plot, shrinkA=0, shrinkB=0)
            # shrinkA=0, shrinkB=0 helps to connect arrows by its tails
            #ax.annotate('', xy=(x_max, y_min), xytext=(x_min, y_min), arrowprops=arrowprops)  # x-axis
            #ax.annotate('', xy=(x_min,y_max), xytext=(x_min, y_min), arrowprops=arrowprops)  # y-axis
            # Draw x-axis arrow (from left to right)
            ax.annotate('', xy=(1.05, 0), xytext=(0, 0), xycoords='axes fraction', arrowprops=arrowprops)
            # Draw y-axis arrow (from bottom to top)
            ax.annotate('', xy=(0, 1.05), xytext=(0, 0), xycoords='axes fraction', arrowprops=arrowprops)

        ax.tick_params(colors=self.text_color_plot)
        ax.xaxis.label.set_color(self.text_color_plot)
        ax.set_facecolor(self.background_color_plot)

        for label in ax.get_xticklabels():
            label.set_fontsize(10)

        for label in ax.get_yticklabels():
            label.set_fontsize(10)

        ax.set_title("X-ray Setup", color=self.text_color_plot, fontsize=10)
    
        fig.patch.set_facecolor(self.background_color_images)
 
        # Show the plot
        #plt.show()

        return fig
    
    def plot_setup_GBI(self, dict_params):
        """
        Plots a schematic of the Structured Grating-based Imaging (GBI) X-ray setup.

        Args:
            dict_params (dict): Dictionary of setup parameters.

        Returns:
            matplotlib.figure.Figure: The figure object containing the setup schematic.
        """
       
        self.type_of_source = dict_params["Source geometry"]
        self.binning_factor = int(float(dict_params["Binning factor"]))
        self.sim_pixel_m = float(dict_params["Sim. pixel (μm)"])*1e-6/self.binning_factor
        self.d_source_det = float(dict_params["Source-Detector distance (m)"])
        self.d_source_grat = float(dict_params["Source-Grating distance (m)"])
        self.d_source_samp = self.d_source_grat + float(dict_params["Grating-Sample distance (m)"])
        self.d_grat_det = self.d_source_det - self.d_source_grat
        self.d_samp_det = self.d_source_det - self.d_source_samp
        self.d_grat_samp = np.abs(self.d_source_samp - self.d_source_grat)
        self.d_prop = self.d_samp_det - (float(dict_params["Whole sample thickness (mm)"])*1e-3)/2
        lightorange = to_rgba("orange", alpha=0.8)

         # Create a figure and axis
        fig, ax = plt.subplots()
 
        # Create a Rectangle object
        source = Rectangle((-0.03, -0.10), width=0.03, height=0.3, linewidth=2, edgecolor='blue', facecolor='lightblue')
        detector = Rectangle((self.d_source_det, -0.3), width=0.01, height=0.6, linewidth=2, edgecolor='green',  facecolor='lightgreen')
        grating = Rectangle((self.d_source_grat-0.01/2, -0.3), width=0.01, height=0.6, linewidth=2, edgecolor='orange',  facecolor=lightorange)
        sample = Rectangle((self.d_source_samp-(float(dict_params["Whole sample thickness (mm)"])*1e-3)/2, -0.3), width=float(dict_params["Whole sample thickness (mm)"])*1e-3, height=0.6, linewidth=2,  edgecolor='red', facecolor='pink')
 
        # Create a Rectangle object legend
        source_legend = Rectangle((0.5*self.d_source_det/8, -0.7), width=self.d_source_det/8, height=0.03, linewidth=2,  edgecolor='blue', facecolor='lightblue')
        detector_legend = Rectangle((2.5*self.d_source_det/8, -0.7), width=self.d_source_det/8, height=0.03, linewidth=2,  edgecolor='green', facecolor='lightgreen')
        grating_legend = Rectangle((4.5*self.d_source_det/8, -0.7), width=self.d_source_det/8, height=0.03, linewidth=2,  edgecolor='orange', facecolor=lightorange)
        sample_legend = Rectangle((6.5*self.d_source_det/8, -0.7), width=self.d_source_det/8, height=0.03, linewidth=2, edgecolor='red',  facecolor='pink')
 
        # Add the rectangle to the plot
        ax.add_patch(source)
        ax.add_patch(detector)
        ax.add_patch(grating)
        ax.add_patch(sample)
 
        ax.add_patch(source_legend)
        ax.add_patch(detector_legend)
        ax.add_patch(grating_legend)
        ax.add_patch(sample_legend)
 
        ax.text(1*self.d_source_det/8, -0.76, 'Source', color='lightblue', ha='center', va='center', fontsize=10)
        ax.text(3*self.d_source_det/8, -0.76, 'Detector', color='lightgreen', ha='center', va='center', fontsize=10)
        ax.text(5*self.d_source_det/8, -0.76, "Grating" , color=lightorange, ha='center', va='center', fontsize=10)
        ax.text(7*self.d_source_det/8, -0.76, "Sample" , color='pink', ha='center', va='center', fontsize=10)
 
        # Set the limits of the plot
        ax.set_xlim(-0.15, self.d_source_det + 0.15)
        ax.set_ylim(-1, 1)
 
        ax.set_xlabel("Propagation axis (m)")
 
        ax.set_yticks([])

        if(arrows_instead_frame == False):
            for spine in ax.spines.values():
                spine.set_linewidth(0.75)  # Set to desired width in pts
                spine.set_edgecolor(self.background_color_images)
                spine.set_color(self.text_color_plot)
        else:
            # Hide the default spines
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(False)
            
            # Keep ticks visible
            ax.tick_params(axis='both', which='both',
                           direction='out',  # or 'inout' as you prefer
                           length=4, width=1,
                           bottom=True, top=False, left=True, right=False)

            
            # Get limits for arrow placement
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            # Add arrows for x and y axes
            arrowprops = dict(arrowstyle='->', linewidth=0.75, color=self.text_color_plot, shrinkA=0, shrinkB=0)
            # shrinkA=0, shrinkB=0 helps to connect arrows by its tails
            #ax.annotate('', xy=(x_max, y_min), xytext=(x_min, y_min), arrowprops=arrowprops)  # x-axis
            #ax.annotate('', xy=(x_min,y_max), xytext=(x_min, y_min), arrowprops=arrowprops)  # y-axis
            # Draw x-axis arrow (from left to right)
            ax.annotate('', xy=(1.05, 0), xytext=(0, 0), xycoords='axes fraction', arrowprops=arrowprops)
            # Draw y-axis arrow (from bottom to top)
            ax.annotate('', xy=(0, 1.05), xytext=(0, 0), xycoords='axes fraction', arrowprops=arrowprops)

        ax.tick_params(colors=self.text_color_plot)
        ax.xaxis.label.set_color(self.text_color_plot)
        ax.set_facecolor(self.background_color_plot)

        for label in ax.get_xticklabels():
            label.set_fontsize(10)

        for label in ax.get_yticklabels():
            label.set_fontsize(10)

        ax.set_title("X-ray Setup", color=self.text_color_plot, fontsize=10)
    
        fig.patch.set_facecolor(self.background_color_images)
 
        # Show the plot
        #plt.show()

        return fig
    
    def plot_setup_inline(self, dict_params):
        """
        Plots a schematic of the Inline X-ray setup.

        Args:
            dict_params (dict): Dictionary of setup parameters.

        Returns:
            matplotlib.figure.Figure: The figure object containing the setup schematic.
        """
       
        self.type_of_source = dict_params["Source geometry"]
        self.binning_factor = int(float(dict_params["Binning factor"]))
        self.sim_pixel_m = float(dict_params["Sim. pixel (μm)"])*1e-6/self.binning_factor
        self.d_source_det = float(dict_params["Source-Detector distance (m)"])
        self.d_source_samp = float(dict_params["Source-Sample distance (m)"])
        self.d_samp_det = self.d_source_det - self.d_source_samp
        self.d_prop = self.d_samp_det - (float(dict_params["Whole sample thickness (mm)"])*1e-3)/2
        lightorange = to_rgba("orange", alpha=0.8)

         # Create a figure and axis
        fig, ax = plt.subplots()
 
        # Create a Rectangle object
        source = Rectangle((-0.03, -0.10), width=0.03, height=0.3, linewidth=2, edgecolor='blue', facecolor='lightblue')
        detector = Rectangle((self.d_source_det, -0.3), width=0.01, height=0.6, linewidth=2, edgecolor='green',  facecolor='lightgreen')
        sample = Rectangle((self.d_source_samp-(float(dict_params["Whole sample thickness (mm)"])*1e-3)/2, -0.3), width=float(dict_params["Whole sample thickness (mm)"])*1e-3, height=0.6, linewidth=2,  edgecolor='red', facecolor='pink')
 
        # Create a Rectangle object legend
        source_legend = Rectangle((0.5*self.d_source_det/6, -0.7), width=self.d_source_det/6, height=0.03, linewidth=2,  edgecolor='blue', facecolor='lightblue')
        detector_legend = Rectangle((2.5*self.d_source_det/6, -0.7), width=self.d_source_det/6, height=0.03, linewidth=2,  edgecolor='green', facecolor='lightgreen')
        sample_legend = Rectangle((4.5*self.d_source_det/6, -0.7), width=self.d_source_det/6, height=0.03, linewidth=2, edgecolor='red',  facecolor='pink')
 
        # Add the rectangle to the plot
        ax.add_patch(source)
        ax.add_patch(detector)
        ax.add_patch(sample)
 
        ax.add_patch(source_legend)
        ax.add_patch(detector_legend)
        ax.add_patch(sample_legend)
 
        ax.text(1*self.d_source_det/6, -0.76, 'Source', color='lightblue', ha='center', va='center', fontsize=10)
        ax.text(3*self.d_source_det/6, -0.76, 'Detector', color='lightgreen', ha='center', va='center', fontsize=10)
        ax.text(5*self.d_source_det/6, -0.76, "Sample" , color='pink', ha='center', va='center', fontsize=10)
 
        # Set the limits of the plot
        ax.set_xlim(-0.15, self.d_source_det + 0.15)
        ax.set_ylim(-1, 1)
 
        ax.set_xlabel("Propagation axis (m)")
 
        ax.set_yticks([])

        if(arrows_instead_frame == False):
            for spine in ax.spines.values():
                spine.set_linewidth(0.75)  # Set to desired width in pts
                spine.set_edgecolor(self.background_color_images)
                spine.set_color(self.text_color_plot)
        else:
            # Hide the default spines
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(False)
            
            # Keep ticks visible
            ax.tick_params(axis='both', which='both',
                           direction='out',  # or 'inout' as you prefer
                           length=4, width=1,
                           bottom=True, top=False, left=True, right=False)

            
            # Get limits for arrow placement
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            # Add arrows for x and y axes
            arrowprops = dict(arrowstyle='->', linewidth=0.75, color=self.text_color_plot, shrinkA=0, shrinkB=0)
            # shrinkA=0, shrinkB=0 helps to connect arrows by its tails
            #ax.annotate('', xy=(x_max, y_min), xytext=(x_min, y_min), arrowprops=arrowprops)  # x-axis
            #ax.annotate('', xy=(x_min,y_max), xytext=(x_min, y_min), arrowprops=arrowprops)  # y-axis
            # Draw x-axis arrow (from left to right)
            ax.annotate('', xy=(1.05, 0), xytext=(0, 0), xycoords='axes fraction', arrowprops=arrowprops)

        ax.tick_params(colors=self.text_color_plot)
        ax.xaxis.label.set_color(self.text_color_plot)
        ax.set_facecolor(self.background_color_plot)

        for label in ax.get_xticklabels():
            label.set_fontsize(10)

        for label in ax.get_yticklabels():
            label.set_fontsize(10)

        ax.set_title("X-ray Setup", color=self.text_color_plot, fontsize=10)
    
        fig.patch.set_facecolor(self.background_color_images)
 
        # Show the plot
        #plt.show()

        return fig