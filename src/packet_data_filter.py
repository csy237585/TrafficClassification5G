from scapy.all import *

def extract_packet_info(packet):
    eth_src = packet.src
    eth_dst = packet.dst
    eth_type = packet.type

    if packet.haslayer(IP):
        ip_src = packet[IP].src
        ip_dst = packet[IP].dst
        ip_proto = packet[IP].proto
        ip_ttl = packet[IP].ttl

        if packet.haslayer(TCP):
            tcp_sport = packet[TCP].sport
            tcp_dport = packet[TCP].dport
            tcp_flags = packet[TCP].flags
            tcp_payload_len = len(packet[TCP].payload)
            tcp_payload = packet[TCP].payload.load if tcp_payload_len > 0 else None
            tcp_seq = packet[TCP].seq
            tcp_ack = packet[TCP].ack
            tcp_window = packet[TCP].window

            return {
                "Ethernet": {"Source MAC": eth_src, "Destination MAC": eth_dst, "Ethernet Type": eth_type},
                "IP": {"Source IP": ip_src, "Destination IP": ip_dst, "IP Protocol": ip_proto,'TTL': ip_ttl},
                "TCP": {"Source Port": tcp_sport, "Destination Port": tcp_dport, "Flags": tcp_flags,
                        "Payload Length": tcp_payload_len, "Payload": tcp_payload, "Sequence Number": tcp_seq,
                       "Acknowledgment Number": tcp_ack, "Window Size": tcp_window}
            }

        elif packet.haslayer(UDP):
            udp_sport = packet[UDP].sport
            udp_dport = packet[UDP].dport
            udp_payload_len = len(packet[UDP].payload)
            udp_payload = packet[UDP].payload.load if udp_payload_len > 0 else None

            return {
                "Ethernet": {"Source MAC": eth_src, "Destination MAC": eth_dst, "Ethernet Type": eth_type},
                "IP": {"Source IP": ip_src, "Destination IP": ip_dst, "IP Protocol": ip_proto,'TTL': ip_ttl},
                "UDP": {"Source Port": udp_sport, "Destination Port": udp_dport,
                        "Payload Length": udp_payload_len, "Payload": udp_payload}
            }

        elif packet.haslayer(ICMP):
            icmp_type = packet[ICMP].type
            icmp_code = packet[ICMP].code
            icmp_payload_len = len(packet[ICMP].payload)
            icmp_payload = packet[ICMP].payload.load if icmp_payload_len > 0 else None

            return {
                "Ethernet": {"Source MAC": eth_src, "Destination MAC": eth_dst, "Ethernet Type": eth_type},
                "IP": {"Source IP": ip_src, "Destination IP": ip_dst, "IP Protocol": ip_proto,'TTL': ip_ttl},
                "ICMP": {"Type": icmp_type, "Code": icmp_code,
                         "Payload Length": icmp_payload_len, "Payload": icmp_payload}
            }

    return None

# Sniff packets
def packet_handler(packet):
    packet_info = extract_packet_info(packet)
    if packet_info:
        print("Packet Info:")
        for layer, info in packet_info.items():
            print(f"{layer}: {info}")
        print("")

# Sniff packets on interface
sniff(prn=packet_handler, store=0)
