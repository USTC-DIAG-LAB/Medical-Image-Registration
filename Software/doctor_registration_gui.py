"""Tkinter interface to the isolated pure-Python registration process."""
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
try:
    import tkinter as tk
    from tkinter import ttk,filedialog,messagebox,scrolledtext,font as tkfont
except ImportError as e:
    raise SystemExit('This Python installation needs Tk/Tkinter for the GUI. The command-line entry is still available.') from e

ROOT=Path(__file__).resolve().parent
PACKAGE_VERSION=(ROOT.parent/'VERSION').read_text(encoding='utf-8').strip()

def build_command(values,threads,samples):
    if threads<1 or samples<50:raise ValueError('Use at least 1 thread and at least 50 sample points.')
    for key in ('moving-image','fixed-image'):
        if not values[key] or not Path(values[key]).is_file():raise ValueError('Select valid moving and fixed image files.')
    if not values['output-dir']:raise ValueError('Select an output folder.')
    out=Path(values['output-dir'])
    if out.exists() and (not out.is_dir() or any(out.iterdir())):raise ValueError('The output folder must be empty. Choose a new folder to preserve existing results.')
    cmd=[sys.executable,'-u',str(ROOT/'one_click_registration.py')]
    for key,value in values.items():cmd+=['--'+key,value]
    return cmd+['--threads',str(threads),'--num-samples',str(samples)]

class App:
    def __init__(self,root):
        self.root=root
        root.title('FOTMIR '+PACKAGE_VERSION+' | Medical Image Registration')
        root.geometry('680x720')
        root.minsize(660,680)
        for name in ('TkDefaultFont','TkTextFont','TkMenuFont'):
            tkfont.nametofont(name).configure(size=16)
        style=ttk.Style(root)
        style.configure('TLabel',font=('Helvetica',16))
        style.configure('TButton',font=('Helvetica',16),padding=(10,6))
        style.configure('TEntry',font=('Helvetica',16),padding=4)
        style.configure('TCombobox',font=('Helvetica',16),padding=3)
        root.option_add('*TCombobox*Listbox.font',('Helvetica',16))
        self.events=queue.Queue();self.process=None;self.busy=False;self.stopping=False;self.stop_time=None;self.closing=False
        self.values={k:tk.StringVar(value=v) for k,v in {'moving-image':'','fixed-image':'','moving-modality':'MRI','fixed-modality':'CT','output-dir':''}.items()}
        self.threads=tk.StringVar(value='1');self.samples=tk.StringVar(value='10000');self.status=tk.StringVar(value='Select input images and an empty output folder.')
        box=ttk.Frame(root,padding=20);box.pack(fill='both',expand=True)
        ttk.Label(box,text='FOTMIR Medical Image Registration',font=('Helvetica',22,'bold')).pack(anchor='w')
        subtitle=ttk.Label(box,text='Register 3D images and save the registered image, six grayscale views, and results summary.',wraplength=620,justify='left')
        subtitle.pack(anchor='w',pady=(8,16))
        form=ttk.Frame(box);form.pack(fill='x');form.columnconfigure(1,weight=1)
        for row,(key,label) in enumerate([('moving-image','Moving image'),('fixed-image','Fixed image'),('output-dir','Output folder')]):
            ttk.Label(form,text=label).grid(row=row,column=0,sticky='w',padx=(0,12),pady=8)
            ttk.Entry(form,textvariable=self.values[key],font=('Helvetica',16),width=20).grid(row=row,column=1,sticky='ew',pady=8)
            ttk.Button(form,text='Browse...',command=lambda k=key:self.choose(k)).grid(row=row,column=2,padx=(10,0))
        opts=ttk.Frame(box);opts.pack(fill='x',pady=(12,8))
        opts.columnconfigure(1,weight=1);opts.columnconfigure(3,weight=1)
        for i,(key,label) in enumerate([('moving-modality','Moving modality'),('fixed-modality','Fixed modality')]):
            ttk.Label(opts,text=label).grid(row=0,column=i*2,sticky='w',padx=(0,10),pady=6)
            ttk.Combobox(opts,textvariable=self.values[key],values=('MRI','CT','PET','SPECT','CBCT'),font=('Helvetica',16),width=7,state='readonly').grid(row=0,column=i*2+1,sticky='ew',padx=(0,16 if i==0 else 0))
        ttk.Label(opts,text='Threads').grid(row=1,column=0,sticky='w',pady=10)
        ttk.Spinbox(opts,from_=1,to=128,textvariable=self.threads,font=('Helvetica',16),width=7).grid(row=1,column=1,sticky='ew',padx=(0,16))
        ttk.Label(opts,text='Sample points').grid(row=1,column=2,sticky='w',padx=(0,10))
        ttk.Entry(opts,textvariable=self.samples,font=('Helvetica',16),width=7).grid(row=1,column=3,sticky='ew')
        actions=ttk.Frame(box);actions.pack(fill='x',pady=5)
        actions.columnconfigure(0,weight=1,uniform='action');actions.columnconfigure(1,weight=1,uniform='action')
        self.run=ttk.Button(actions,text='Start registration',command=self.start);self.run.grid(row=0,column=0,sticky='ew',padx=(0,6),pady=4)
        self.cancel=ttk.Button(actions,text='Cancel',command=self.stop,state='disabled');self.cancel.grid(row=0,column=1,sticky='ew',padx=(6,0),pady=4)
        self.check=ttk.Button(actions,text='Check environment',command=self.check_env);self.check.grid(row=1,column=0,sticky='ew',padx=(0,6),pady=4)
        ttk.Button(actions,text='Open output folder',command=self.open_output).grid(row=1,column=1,sticky='ew',padx=(6,0),pady=4)
        status_label=ttk.Label(box,textvariable=self.status,wraplength=620,justify='left')
        status_label.pack(anchor='w',pady=(10,8))
        self.log=scrolledtext.ScrolledText(box,height=8,width=40,font=('Helvetica',14),wrap='word',state='disabled');self.log.pack(fill='both',expand=True)
        def fit_text(event):
            width=max(200,event.width-40)
            subtitle.configure(wraplength=width);status_label.configure(wraplength=width)
        box.bind('<Configure>',fit_text)
        root.protocol('WM_DELETE_WINDOW',self.close);root.after(100,self.poll)
    def choose(self,key):
        p=filedialog.askdirectory(title='Select an empty output folder') if key=='output-dir' else filedialog.askopenfilename(title='Select a 3D image file')
        if p:self.values[key].set(p)
    def append(self,text):
        self.log.configure(state='normal');self.log.insert('end',text);self.log.see('end');self.log.configure(state='disabled')
    def launch(self,cmd):
        self.busy=True;self.stopping=False;self.run.configure(state='disabled');self.check.configure(state='disabled');self.cancel.configure(state='normal');self.status.set('Running...')
        try:self.process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=(os.name!='nt'))
        except Exception as e:self.events.put(('done',str(e)));return
        def read():
            try:
                for line in self.process.stdout:self.events.put(('log',line))
                code=self.process.wait();self.events.put(('done',code))
            except Exception as e:self.events.put(('done',str(e)))
        threading.Thread(target=read,daemon=True).start()
    def start(self):
        if self.busy:return
        try:cmd=build_command({k:v.get().strip() for k,v in self.values.items()},int(self.threads.get()),int(self.samples.get()))
        except (ValueError,TypeError) as e:messagebox.showerror('Check input',str(e));return
        self.launch(cmd)
    def check_env(self):
        if not self.busy:self.launch([sys.executable,'-u',str(ROOT/'one_click_registration.py'),'--check-environment'])
    def signal_process(self,force=False):
        p=self.process
        if p is None or p.poll() is not None:return
        try:
            if os.name=='nt':subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            else:os.killpg(p.pid,signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:pass
    def stop(self):
        if self.busy:
            self.stopping=True;self.stop_time=time.monotonic();self.status.set('Stopping...');self.signal_process()
    def poll(self):
        try:
            while True:
                kind,item=self.events.get_nowait()
                if kind=='log':self.append(item)
                else:
                    self.busy=False;self.process=None;self.run.configure(state='normal');self.check.configure(state='normal');self.cancel.configure(state='disabled')
                    status='Cancelled. If partial files were saved, use a new empty output folder before retrying.' if self.stopping else ('Completed.' if item==0 else 'Run failed. See the log for details.')
                    self.status.set(status);self.append('\n'+status+'\n')
                    if not isinstance(item,int):self.append(str(item)+'\n')
        except queue.Empty:pass
        if self.stopping and self.busy and time.monotonic()-self.stop_time>5:self.signal_process(force=True)
        if self.closing and not self.busy:self.root.destroy();return
        self.root.after(100,self.poll)
    def close(self):
        if self.busy:
            if not messagebox.askyesno('Exit','Stop the current task and exit?'):return
            self.closing=True;self.stop()
        else:self.root.destroy()
    def open_output(self):
        p=Path(self.values['output-dir'].get())
        if not self.values['output-dir'].get() or not p.is_dir():messagebox.showinfo('Output folder','Select an existing output folder first.');return
        if sys.platform=='darwin':subprocess.Popen(['open',str(p.resolve())])
        elif os.name=='nt':os.startfile(str(p.resolve()))
        else:subprocess.Popen(['xdg-open',str(p.resolve())])

if __name__=='__main__':
    root=tk.Tk();App(root);root.mainloop()
