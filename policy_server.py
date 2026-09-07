import os
import sys
import time
import socket
import pickle
import torch
import numpy as np

from eval_leisaac import DINOv2PolicyConnector

def recv_all(sock, n):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def main():
    checkpoint = "Gogul99/RoboMamba_DINO_ACT_Policy_Real"
    cfg_path = "config.yaml"
    
    
    print("  Starting Custom Policy Server on GPU")
    print("==========================================================")
    connector = DINOv2PolicyConnector(checkpoint_path=checkpoint, cfg_path=cfg_path)
    connector.reset()
    print("==========================================================")
    
    # Create TCP socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("127.0.0.1", 8085))
    server_socket.listen(1)
    print(" Policy Server listening on 127.0.0.1:8085...")

    while True:
        try:
            conn, addr = server_socket.accept()
            print(f"\n Client connected from {addr}")
            while True:
                # Read header (size of packet)
                header = recv_all(conn, 4)
                if not header:
                    break
                data_size = int.from_bytes(header, byteorder="big")
                
                # Receive packet data
                data = b""
                while len(data) < data_size:
                    packet = conn.recv(data_size - len(data))
                    if not packet:
                        break
                    data += packet
                
                if not data:
                    break
                
                # Deserialise request
                req = pickle.loads(data)
                current_t = req["current_t"]
                raw_images = req["raw_images"]
                raw_state = req["raw_state"]
                
                t_start = time.perf_counter()
                
                # Preprocess images on GPU
                processed_imgs = []
                for img in raw_images:
                    img_t = torch.tensor(img, dtype=torch.float32, device=connector.device)
                    if img_t.ndim == 3 and img_t.shape[-1] in [3, 4]:
                        img_t = img_t[..., :3].permute(2, 0, 1)
                    if img_t.shape[-2:] != (224, 224):
                        img_t = torch.nn.functional.interpolate(
                            img_t.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
                        ).squeeze(0)
                    if img_t.max() > 1.0:
                        img_t = img_t / 255.0
                    mean = torch.tensor([0.485, 0.456, 0.406], device=connector.device).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=connector.device).view(3, 1, 1)
                    img_t = (img_t - mean) / std
                    processed_imgs.append(img_t)
                batch_imgs = torch.stack(processed_imgs, dim=0).unsqueeze(0)
                
                # Pass raw state directly to policy (connector.policy normalizes & unnormalizes internally)
                state_t = torch.tensor(raw_state, dtype=torch.float32, device=connector.device).unsqueeze(0)
                
                # Model Inference
                with torch.no_grad():
                    pred_chunk, _, _ = connector.policy(batch_imgs, state_t)
                    pred_chunk = pred_chunk.squeeze(0).cpu().numpy()  # [16, 6] in true physical degrees
                
                inference_dt = (time.perf_counter() - t_start) * 1000.0
                print(f"\rStep {current_t:3d} | GPU Inference: {inference_dt:5.1f}ms", end="", flush=True)
                
                # Serialise response
                resp_data = pickle.dumps(pred_chunk)
                conn.sendall(len(resp_data).to_bytes(4, byteorder="big") + resp_data)
                
        except ConnectionResetError:
            print("\n Client disconnected.")
        except KeyboardInterrupt:
            print("\n Stopping Policy Server safely.")
            break
        except Exception as e:
            print(f"\n Error: {e}")
            break
            
    server_socket.close()

if __name__ == "__main__":
    main()

