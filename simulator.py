import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style

import tkinter as tk
from tkinter import ttk

import json
import threading
from util import *
from collections import OrderedDict
from xmlrpc.server import SimpleXMLRPCServer

LARGE_FONT= ("Verdana", 12)
style.use("ggplot")

report = {}

f = Figure(figsize=(5,5), dpi=100)
a = f.add_subplot(111)

def animate(i):
	pullData = open("sampleText.txt","r").read()
	dataList = pullData.split('\n')
	xList = []
	yList = []
	for eachLine in dataList:
		if len(eachLine) > 1:
			x, y = eachLine.split(',')
			xList.append(int(x))
			yList.append(int(y))

	a.clear()
	a.plot(xList, yList)

class Simulator(tk.Tk):

	def __init__(self, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		
		tk.Tk.wm_title(self, "Simulator for IOT Analytics")
		
		self.container = tk.Frame(self)
		self.container.pack(side="top", fill="both", expand = True)
		self.container.grid_rowconfigure(0, weight=1)
		self.container.grid_columnconfigure(0, weight=1)

		self.frames = {}
		for F in (HomePage,):
			frame = F(self.container, self)
			self.frames[F] = frame
			frame.grid(row=0, column=0, sticky="nsew")

		self.protocol('WM_DELETE_WINDOW', self.quit)
		self.show_frame(HomePage)

	def show_frame(self, cont):
		frame = self.frames[cont]
		frame.tkraise()

	def init_menu(self,data):
		menubar = tk.Menu(self.container)

		nodemenu = tk.Menu(menubar, tearoff=0)
		for node_id in data:
			nodemenu.add_command(label="Node %d"%node_id, command=lambda x=node_id: self.show_frame(x))
		menubar.add_cascade(label="Nodes", menu=nodemenu)
		tk.Tk.config(self, menu=menubar)

	def quit(self):
		self.destroy()
		
class HomePage(tk.Frame):

	def __init__(self, parent, controller):
		tk.Frame.__init__(self,parent)
		self.controller = controller

		label = tk.Label(self, text="Home Page", font=LARGE_FONT)
		label.pack(pady=10,padx=10)

		buttons = OrderedDict()
		buttons["choose_yaml"] = ttk.Button(self, text="Choose yaml",command=self.browse)
		buttons["start_simulation"] = ttk.Button(self, text="Start Simulation",command=self.start_simulation)
		
		self.yaml_view = tk.Text(self, state="disabled", borderwidth=3, relief="sunken")
		self.yaml_view.config(font=("consolas", 12), undo=True, wrap='word')

		scrollb = tk.Scrollbar(self, command=self.yaml_view.yview)
		self.yaml_view['yscrollcommand'] = scrollb.set
		
		self.yaml_view.pack()
		for _,b in buttons.items(): b.pack()

	def browse(self):
		self.config_yaml = tk.filedialog.askopenfilename()
		try:
			yaml = open(self.config_yaml).read()
			self.yaml_view.configure(state="normal")
			self.yaml_view.insert(tk.END, yaml)
			self.yaml_view.configure(state="disabled")
		except FileNotFoundError:
			pass

	def start_simulation(self):
		own_address = (get_ip(),9000)
		self.controller.data = read_yaml(own_address,self.config_yaml)
		server_thread = threading.Thread(target=start_server,args=(own_address,))
		server_thread.start()
		trigger_scripts()

		for node_id in self.controller.data:
			frame = NodePage(self.controller.container, self.controller, node_id)
			self.controller.frames[node_id] = frame
			frame.grid(row=0, column=0, sticky="nsew")

		self.controller.init_menu(self.controller.data)
		self.controller.show_frame(1)
		# server_thread.join()

class NodePage(tk.Frame):

	def __init__(self, parent, controller,node_id):
		tk.Frame.__init__(self, parent)
		self.controller = controller
		self.node_id = node_id

		label = tk.Label(self, text="Node %d"%node_id, font=LARGE_FONT)
		label.pack(pady=10,padx=10)

		self.views = OrderedDict({"CONN":None,"STAT":None})
		for _type in self.views:
			self.views[_type] = tk.Text(self, state="disabled", borderwidth=3, relief="sunken")
			self.views[_type].config(font=("consolas", 12), undo=True, wrap='word')

			scrollb = tk.Scrollbar(self, command=self.views[_type].yview)
			self.views[_type]['yscrollcommand'] = scrollb.set
			
			self.views[_type].pack()

		# canvas = FigureCanvasTkAgg(f, self)
		# canvas.show()
		# canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

		# toolbar = NavigationToolbar2TkAgg(canvas, self)
		# toolbar.update()
		# canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

	def add_log(self,_type,log):
		print("here323")
		self.views[_type].configure(state="normal")
		self.views[_type].insert(tk.END, log)
		self.views[_type].configure(state="disabled")


# Start XML RPC server on the Master machine
def start_server(own_address):
	print('RPC server started at http://%s:%d'%own_address)
	server = SimpleXMLRPCServer(own_address, allow_none=True)
	server.register_function(log_report)
	server.register_function(shutdown_thread)
	server.serve_forever()

def remote_shutdown():
	print("alskdaskdlask")
	t = threading.Thread(target=shutdown_thread)
	t.start();t.join()

def shutdown_thread():
	server.shutdown()

# Get logs from nodes
def log_report(log):
	print(log)
	log = json.loads(log)
	if __name__ == '__main__':
		log = json.loads(log)
		if log["node_id"] not in report:
			report[log["node_id"]] = {}

		if log["type"] not in report[log["node_id"]]:
			report[log["node_id"]][log["type"]] = []

		report[log["node_id"]][log["type"]].append(log["payload"])
		print(log)
		# simulator.frame[log["node_id"]].add_log(log["type"],str(log["payload"]))

if __name__ == '__main__':
	simulator = Simulator()
	# ani = animation.FuncAnimation(f, animate, interval=1000)
	simulator.mainloop()