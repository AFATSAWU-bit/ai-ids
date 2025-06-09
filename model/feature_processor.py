import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

class FeatureProcessor:
    def __init__(self):
        self.required_features = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'flow_duration', 'flow_bytes_per_sec',
            'flow_packets_per_sec', 'min_packet_length', 'max_packet_length',
            'packet_length_mean', 'packet_length_std', 'packet_length_variance',
            'packet_length_skewness', 'packet_length_kurtosis', 'fwd_packet_length_max',
            'fwd_packet_length_min', 'fwd_packet_length_mean', 'fwd_packet_length_std',
            'fwd_packet_length_variance', 'fwd_packet_length_skewness',
            'fwd_packet_length_kurtosis', 'bwd_packet_length_max', 'bwd_packet_length_min',
            'bwd_packet_length_mean', 'bwd_packet_length_std', 'bwd_packet_length_variance',
            'bwd_packet_length_skewness', 'bwd_packet_length_kurtosis', 'min_inter_arrival_time',
            'max_inter_arrival_time', 'inter_arrival_time_mean', 'inter_arrival_time_std',
            'inter_arrival_time_variance', 'inter_arrival_time_skewness',
            'inter_arrival_time_kurtosis', 'active_mean', 'active_std', 'active_max',
            'active_min', 'idle_mean', 'idle_std', 'idle_max', 'idle_min',
            'fwd_header_length', 'bwd_header_length', 'fwd_seg_size_min',
            'fwd_seg_size_max', 'fwd_seg_size_mean', 'fwd_seg_size_std',
            'fwd_seg_size_variance', 'bwd_seg_size_min', 'bwd_seg_size_max',
            'bwd_seg_size_mean', 'bwd_seg_size_std', 'bwd_seg_size_variance',
            'init_win_bytes_forward', 'init_win_bytes_backward', 'act_data_pkt_fwd',
            'min_seg_size_forward'
        ]
        self.feature_imputation = {
            feature: 0 for feature in self.required_features
        }
        self.scaler = None
        
    def load_scaler(self, scaler_path):
        self.scaler = joblib.load(scaler_path)
        
    def preprocess(self, df):
        """Handle flexible input features"""
        # 1. Convert all columns to numeric where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 2. Add missing columns with default values
        for feature in self.required_features:
            if feature not in df.columns:
                df[feature] = self.feature_imputation.get(feature, 0)
        
        # 3. Keep only the required features in correct order
        df = df[self.required_features]
        
        # 4. Handle infinite/NaN values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        return df

    def scale_features(self, df):
        if not self.scaler:
            raise ValueError("Scaler not loaded")
        return self.scaler.transform(df)
