variable "region" {
  default = "us-east-1"
}

variable "app_name" {
  default = "Lightweight ML Serving API"
}

variable "app_zip_path" {
  description = "Path to your zipped application bundle"
  type        = string
}
