import time
from server.fedpars import FedPARS
from options import args_parser
from model.protounet import ProtoUnet

def run(args):
    if args.algorithm == "FedPARS":
        model = ProtoUnet(num_prototype=args.num_prototype,gamma_list=args.gamma_list).to(args.device)
        server = FedPARS(args,model)

    server.train()

if __name__ == "__main__":
    total_start = time.time()
    args = args_parser()
    run(args)

